"""6軸アーム+ハンドをviserで表示しつつ、FastAPI経由で関節角度を送れるサーバー。

アームとハンドの合成方法は 002_arm_with_hand.py と同じで、アーム手先リンク(既定 tool0)に
中継フレームを立ててハンドをぶら下げる。腕の関節角度が変わるたびに、その中継フレームの姿勢も
FKで計算し直してハンドを追従させる。

viserの3Dビューア(デフォルト http://localhost:8080)とは別に、
関節角度を受け取るHTTP API(デフォルト http://localhost:8000)を同じプロセスで立てる。

    python src/003_server_with_api.py
    python src/003_server_with_api.py --arm assets/arm/arm.urdf --hand assets/hand/hand_jig.urdf --http-port 8000

API利用例:
    # 6軸分の角度をまとめて送る(順序はアームURDFの可動関節の定義順)
    curl -X POST http://localhost:8000/joints ^
        -H "Content-Type: application/json" ^
        -d "{\"angles\": [0.3, -0.2, 0, 0, 0, 0]}"

    # サーバー側にあるCSVのパスを渡して、時系列の角度軌道を再生させる
    # (CSVの形式は TRAJECTORY_CSV_FORMAT を参照)
    curl -X POST http://localhost:8000/trajectory ^
        -H "Content-Type: application/json" ^
        -d "{\"csv_path\": \"C:/path/to/trajectory.csv\"}"
"""

from __future__ import annotations

import argparse
import csv
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
import uvicorn
import viser
import viser.extras
import viser.transforms as vtf
import yourdfpy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

TRAJECTORY_CSV_FORMAT = """
先頭列が "t"(秒、単調増加)、続く6列が関節角度(アームURDFの可動関節の定義順)のヘッダ付きCSV。
例:
    t,joint1,joint2,joint3,joint4,joint5,joint6
    0.0,0,0,0,0,0,0
    0.1,0.05,0,0,0,0,0
"""

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_ARM_URDF = ASSETS_DIR / "arm" / "arm.urdf"
DEFAULT_HAND_URDF = ASSETS_DIR / "hand" / "hand_jig.urdf"

# ハンドを取り付けるアーム側リンク名と、ハンドを吊るすシーンノード名。固定値なので変数化しない。
MOUNT_LINK = "tool0"
MOUNT_NODE_NAME = "/tool_mount"

# このサーバーは6軸アーム専用。URDFの可動関節数がこれと一致しない場合は起動時に弾く。
NUM_JOINTS = 6


class AnglesRequest(BaseModel):
    # 6軸固定なので、常にちょうど6個の角度をまとめて指定する。
    angles: list[float] = Field(min_length=NUM_JOINTS, max_length=NUM_JOINTS)


class TrajectoryRequest(BaseModel):
    # クライアントはファイル本体ではなく、サーバーから見えるCSVのパスを送る。
    csv_path: str


def load_urdf(path: Path) -> yourdfpy.URDF:
    """メッシュの相対パスをURDFのある位置から解決して読み込む。"""
    return yourdfpy.URDF.load(
        str(path),
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=str(path.parent)),
    )


def add_arm_with_hand(
    server: viser.ViserServer,
    arm_urdf_path: Path,
    hand_urdf_path: Path,
) -> tuple[viser.extras.ViserUrdf, yourdfpy.URDF, viser.FrameHandle]:
    """アームとハンドのURDFを読み込み、ハンドをアーム手先(MOUNT_LINK)に取り付けてシーンに追加する。

    Returns:
        (アームのViserUrdf, アームのyourdfpyモデル, ハンドを吊るす中継フレーム)。
    """
    # 順運動学を自前で参照したいので yourdfpy のモデルを保持しておく
    # (ViserUrdf.update_cfg はこのオブジェクトをそのまま更新してくれる)。
    arm_model = load_urdf(arm_urdf_path)
    arm_urdf = viser.extras.ViserUrdf(server, urdf_or_path=arm_model)

    # ハンドは別URDFなので中継フレームを親にしてぶら下げる。
    tool_frame = server.scene.add_frame(MOUNT_NODE_NAME, show_axes=False)
    hand_urdf = viser.extras.ViserUrdf(
        server, urdf_or_path=load_urdf(hand_urdf_path), root_node_name=MOUNT_NODE_NAME
    )
    # ハンド側に可動関節はない想定だが、固定ジョイントの姿勢を一度反映しておく。
    hand_urdf.update_cfg(np.zeros(len(hand_urdf.get_actuated_joint_names())))

    # 全関節0の姿勢で表示し、中継フレームをそのときの手先姿勢に合わせる。
    arm_urdf.update_cfg(np.zeros(len(arm_urdf.get_actuated_joint_names())))
    sync_tool_frame(arm_model, tool_frame)

    return arm_urdf, arm_model, tool_frame


def sync_tool_frame(arm_model: yourdfpy.URDF, tool_frame: viser.FrameHandle) -> None:
    """アームの現在姿勢(FK)に合わせて、ハンドを吊るす中継フレームを追従させる。"""
    T_world_mount = arm_model.get_transform(MOUNT_LINK)
    tool_frame.wxyz = vtf.SO3.from_matrix(T_world_mount[:3, :3]).wxyz
    tool_frame.position = T_world_mount[:3, 3]


def load_trajectory(csv_path: Path) -> tuple[list[float], list[list[float]]]:
    """時系列角度軌道のCSVを読み込む。形式は TRAJECTORY_CSV_FORMAT を参照。"""
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]

    if not rows:
        raise ValueError(f"CSVが空です: {csv_path}")

    header, data_rows = rows[0], rows[1:]
    if header[0].strip().lower() != "t" or len(header) != NUM_JOINTS + 1:
        raise ValueError(
            "CSVのヘッダが不正です。先頭列は't'、続けて"
            f"{NUM_JOINTS}列の関節角度が必要です: {header}"
        )
    if not data_rows:
        raise ValueError(f"CSVにデータ行がありません: {csv_path}")

    times = [float(row[0]) for row in data_rows]
    angles_list = [[float(v) for v in row[1:]] for row in data_rows]
    if times != sorted(times):
        raise ValueError("CSVの't'列は単調増加している必要があります")

    return times, angles_list


class TrajectoryPlayer:
    """時系列角度軌道をバックグラウンドスレッドで再生する。

    新しい軌道が来たら、再生中の前の軌道は打ち切って乗り換える。
    """

    def __init__(
        self,
        arm_urdf: viser.extras.ViserUrdf,
        arm_model: yourdfpy.URDF,
        tool_frame: viser.FrameHandle,
    ) -> None:
        self._arm_urdf = arm_urdf
        self._arm_model = arm_model
        self._tool_frame = tool_frame
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

    def play(self, times: list[float], angles_list: list[list[float]]) -> None:
        # 前の再生が動いていれば止めてから新しい軌道を始める。
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        stop_event = threading.Event()
        self._stop_event = stop_event
        self._thread = threading.Thread(
            target=self._run, args=(times, angles_list, stop_event), daemon=True
        )
        self._thread.start()

    def _run(self, times: list[float], angles_list: list[list[float]], stop_event: threading.Event) -> None:
        start_wall = time.monotonic()
        start_t = times[0]
        for t, angles in zip(times, angles_list):
            if stop_event.is_set():
                return
            wait_sec = (t - start_t) - (time.monotonic() - start_wall)
            if wait_sec > 0 and stop_event.wait(wait_sec):
                return
            self._arm_urdf.update_cfg(np.array(angles, dtype=float))
            # 腕が動くとハンドの取り付け位置(手先姿勢)も変わるので追従させる。
            sync_tool_frame(self._arm_model, self._tool_frame)


def create_app(
    arm_urdf: viser.extras.ViserUrdf,
    arm_model: yourdfpy.URDF,
    tool_frame: viser.FrameHandle,
) -> FastAPI:
    """角度を受け取ってviserの表示姿勢に反映するだけのFastAPIアプリを作る。

    取得(GET)や現在角度の保持は行わない、送りっぱなしのSET専用API。
    """
    app = FastAPI(title="ViserServer joint API")
    player = TrajectoryPlayer(arm_urdf, arm_model, tool_frame)

    @app.post("/joints")
    def post_joints(req: AnglesRequest) -> dict:
        arm_urdf.update_cfg(np.array(req.angles, dtype=float))
        # 腕が動くとハンドの取り付け位置(手先姿勢)も変わるので追従させる。
        sync_tool_frame(arm_model, tool_frame)
        return {"ok": True}

    @app.post("/trajectory")
    def post_trajectory(req: TrajectoryRequest) -> dict:
        csv_path = Path(req.csv_path)
        if not csv_path.is_file():
            raise HTTPException(status_code=404, detail=f"CSVが見つかりません: {csv_path}")
        try:
            times, angles_list = load_trajectory(csv_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        player.play(times, angles_list)
        return {"ok": True, "num_points": len(times), "duration_sec": times[-1] - times[0]}

    return app


def main(
    arm_urdf_path: Path,
    hand_urdf_path: Path,
    viser_port: int,
    http_port: int,
) -> None:
    server = viser.ViserServer(port=viser_port)
    server.scene.add_grid("/grid", width=2.0, height=2.0)

    arm_urdf, arm_model, tool_frame = add_arm_with_hand(server, arm_urdf_path, hand_urdf_path)
    joint_names = arm_urdf.get_actuated_joint_names()
    if len(joint_names) != NUM_JOINTS:
        raise ValueError(
            f"このサーバーは{NUM_JOINTS}軸のURDF専用です。"
            f"実際の可動関節数: {len(joint_names)} ({joint_names})"
        )

    print(f"Open your browser to http://localhost:{viser_port}")
    print(f"Joint API listening on http://localhost:{http_port} (POST /joints, POST /trajectory)")
    print("Press Ctrl+C to exit")

    app = create_app(arm_urdf, arm_model, tool_frame)
    # viserは内部で自前のサーバースレッドを立てて非同期に動くので、
    # ここでuvicornをフォアグラウンドで走らせてプロセスを維持する。
    uvicorn.run(app, host="0.0.0.0", port=http_port, log_level="info")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--arm", type=Path, default=DEFAULT_ARM_URDF, help="アームのURDFファイル(6軸限定)")
    parser.add_argument("--hand", type=Path, default=DEFAULT_HAND_URDF, help="ハンドのURDFファイル")
    parser.add_argument("--viser-port", type=int, default=8080, help="viser 3DビューアのHTTPポート")
    parser.add_argument("--http-port", type=int, default=8000, help="関節角度APIのHTTPポート")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.arm, args.hand, viser_port=args.viser_port, http_port=args.http_port)
