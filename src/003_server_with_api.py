"""viserでURDFロボットを表示しつつ、FastAPI経由でクライアントから関節角度を送れるサーバー。

viserの3Dビューア(デフォルト http://localhost:8080)とは別に、
関節角度を受け取るHTTP API(デフォルト http://localhost:8000)を同じプロセスで立てる。
クライアントはこのAPIにPOSTするだけで、ブラウザ上のロボット姿勢をリアルタイムに動かせる。

    python src/003_server_with_api.py
    python src/003_server_with_api.py --urdf assets/arm/arm.urdf --http-port 8000

クライアント側の例:
    # 関節名と現在の角度を取得
    curl http://localhost:8000/joints

    # 全関節の角度をまとめて送る(順序は /joints の names に合わせる)
    curl -X POST http://localhost:8000/joints ^
        -H "Content-Type: application/json" ^
        -d "{\"angles\": [0.3, -0.2, 0, 0, 0, 0]}"

    # 1関節だけ送る
    curl -X POST http://localhost:8000/joints/tool0 ^
        -H "Content-Type: application/json" ^
        -d "{\"angle\": 0.5}"
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import numpy as np
import uvicorn
import viser
import viser.extras
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_URDF = ASSETS_DIR / "arm" / "arm.urdf"


class AnglesRequest(BaseModel):
    angles: list[float]


class AngleRequest(BaseModel):
    angle: float


class JointState:
    """viserのURDF姿勢と関節角度を、複数スレッドから安全にやり取りするための状態管理。

    FastAPI(uvicorn)は自前のスレッドでリクエストを処理し、viserサーバーも
    バックグラウンドスレッドで動いているため、角度の読み書きはロックで保護する。
    """

    def __init__(self, viser_urdf: viser.extras.ViserUrdf):
        self._viser_urdf = viser_urdf
        self._names = list(viser_urdf.get_actuated_joint_names())
        self._angles = np.zeros(len(self._names))
        self._lock = threading.Lock()
        # 初期姿勢(全関節0)を反映しておく。
        viser_urdf.update_cfg(self._angles)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def get_angles(self) -> list[float]:
        with self._lock:
            return self._angles.tolist()

    def set_all(self, angles: list[float]) -> None:
        if len(angles) != len(self._names):
            raise ValueError(
                f"angles の長さが関節数({len(self._names)})と一致しません: {len(angles)}"
            )
        with self._lock:
            self._angles = np.array(angles, dtype=float)
            self._viser_urdf.update_cfg(self._angles)

    def set_one(self, name: str, angle: float) -> None:
        with self._lock:
            if name not in self._names:
                raise KeyError(name)
            self._angles[self._names.index(name)] = angle
            self._viser_urdf.update_cfg(self._angles)


def create_app(state: JointState) -> FastAPI:
    app = FastAPI(title="ViserServer joint API")
    # ブラウザ上の別オリジンJSクライアントなどからも叩けるようにしておく。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/joints")
    def get_joints() -> dict:
        return {"names": state.names, "angles": state.get_angles()}

    @app.post("/joints")
    def post_joints(req: AnglesRequest) -> dict:
        try:
            state.set_all(req.angles)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"names": state.names, "angles": state.get_angles()}

    @app.post("/joints/{name}")
    def post_joint(name: str, req: AngleRequest) -> dict:
        try:
            state.set_one(name, req.angle)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown joint: {name}")
        return {"names": state.names, "angles": state.get_angles()}

    return app


def main(urdf_path: Path, viser_port: int, http_port: int) -> None:
    server = viser.ViserServer(port=viser_port)
    server.scene.add_grid("/grid", width=2.0, height=2.0)

    viser_urdf = viser.extras.ViserUrdf(server, urdf_or_path=urdf_path)
    state = JointState(viser_urdf)

    print(f"Open your browser to http://localhost:{viser_port}")
    print(f"Joint API listening on http://localhost:{http_port} (GET/POST /joints)")
    print("Press Ctrl+C to exit")

    app = create_app(state)
    # viserは内部で自前のサーバースレッドを立てて非同期に動くので、
    # ここでuvicornをフォアグラウンドで走らせてプロセスを維持する。
    uvicorn.run(app, host="0.0.0.0", port=http_port, log_level="info")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="表示するURDFファイル")
    parser.add_argument("--viser-port", type=int, default=8080, help="viser 3DビューアのHTTPポート")
    parser.add_argument("--http-port", type=int, default=8000, help="関節角度APIのHTTPポート")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.urdf, viser_port=args.viser_port, http_port=args.http_port)
