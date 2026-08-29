"""6軸アームをviserで表示しつつ、FastAPI経由で全軸の関節角度をまとめて送れるサーバー。

viserの3Dビューア(デフォルト http://localhost:8080)とは別に、
関節角度を受け取るHTTP API(デフォルト http://localhost:8000)を同じプロセスで立てる。
6軸分の角度を毎回まとめて送る形のみをサポートする(1軸だけの部分更新や、現在角度の取得はしない)。

    python src/003_server_with_api.py
    python src/003_server_with_api.py --urdf assets/arm/arm.urdf --http-port 8000

API利用例:
    # 6軸分の角度をまとめて送る(順序はURDFの可動関節の定義順)
    curl -X POST http://localhost:8000/joints ^
        -H "Content-Type: application/json" ^
        -d "{\"angles\": [0.3, -0.2, 0, 0, 0, 0]}"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import uvicorn
import viser
import viser.extras
from fastapi import FastAPI
from pydantic import BaseModel, Field

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_URDF = ASSETS_DIR / "arm" / "arm.urdf"

# このサーバーは6軸アーム専用。URDFの可動関節数がこれと一致しない場合は起動時に弾く。
NUM_JOINTS = 6


class AnglesRequest(BaseModel):
    # 6軸固定なので、常にちょうど6個の角度をまとめて指定する。
    angles: list[float] = Field(min_length=NUM_JOINTS, max_length=NUM_JOINTS)


def create_app(viser_urdf: viser.extras.ViserUrdf) -> FastAPI:
    """角度を受け取ってviserの表示姿勢に反映するだけのFastAPIアプリを作る。

    取得(GET)や現在角度の保持は行わない、送りっぱなしのSET専用API。
    """
    app = FastAPI(title="ViserServer joint API")

    @app.post("/joints")
    def post_joints(req: AnglesRequest) -> dict:
        viser_urdf.update_cfg(np.array(req.angles, dtype=float))
        return {"ok": True}

    return app


def main(urdf_path: Path, viser_port: int, http_port: int) -> None:
    server = viser.ViserServer(port=viser_port)
    server.scene.add_grid("/grid", width=2.0, height=2.0)

    viser_urdf = viser.extras.ViserUrdf(server, urdf_or_path=urdf_path)
    joint_names = viser_urdf.get_actuated_joint_names()
    if len(joint_names) != NUM_JOINTS:
        raise ValueError(
            f"このサーバーは{NUM_JOINTS}軸のURDF専用です。"
            f"実際の可動関節数: {len(joint_names)} ({joint_names})"
        )
    # 初期姿勢(全関節0)を反映しておく。
    viser_urdf.update_cfg(np.zeros(NUM_JOINTS))

    print(f"Open your browser to http://localhost:{viser_port}")
    print(f"Joint API listening on http://localhost:{http_port} (POST /joints)")
    print("Press Ctrl+C to exit")

    app = create_app(viser_urdf)
    # viserは内部で自前のサーバースレッドを立てて非同期に動くので、
    # ここでuvicornをフォアグラウンドで走らせてプロセスを維持する。
    uvicorn.run(app, host="0.0.0.0", port=http_port, log_level="info")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="表示するURDFファイル(6軸限定)")
    parser.add_argument("--viser-port", type=int, default=8080, help="viser 3DビューアのHTTPポート")
    parser.add_argument("--http-port", type=int, default=8000, help="関節角度APIのHTTPポート")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.urdf, viser_port=args.viser_port, http_port=args.http_port)
