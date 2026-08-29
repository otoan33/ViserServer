"""アームURDFとハンドURDFを別ファイルのまま Viser で合成表示する。

アーム(例: assets/arm/arm.urdf)とハンド(例: assets/hand/hand_jig.urdf)は
独立したURDFで、アーム手先リンクに相当する中継フレームを1つ立てて
そこにハンドをぶら下げることで合成する。

    python src/002_arm_with_hand.py
    python src/002_arm_with_hand.py --arm path/to/arm.urdf --hand path/to/hand.urdf
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np
import viser
import viser.extras
import viser.transforms as vtf
import yourdfpy

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_ARM_URDF = ASSETS_DIR / "arm" / "arm.urdf"
DEFAULT_HAND_URDF = ASSETS_DIR / "hand" / "hand_jig.urdf"

# ハンドを取り付けるアーム側リンク名と、ハンドを吊るすシーンノード名。
DEFAULT_MOUNT_LINK = "tool0"
MOUNT_NODE_NAME = "/tool_mount"


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
    mount_link: str = DEFAULT_MOUNT_LINK,
    mount_node_name: str = MOUNT_NODE_NAME,
) -> tuple[viser.extras.ViserUrdf, viser.extras.ViserUrdf, viser.FrameHandle]:
    """アームとハンドのURDFを読み込み、ハンドをアーム手先に取り付けてシーンに追加する。

    Args:
        server: 追加先の ViserServer。
        arm_urdf_path: アームのURDFファイル。
        hand_urdf_path: ハンドのURDFファイル。ルートリンクの原点が mount_link に一致する。
        mount_link: ハンドを取り付けるアーム側のリンク名。
        mount_node_name: ハンドを吊るす中継フレームのシーンノード名。

    Returns:
        (アームのViserUrdf, ハンドのViserUrdf, 中継フレーム)。
    """
    # 順運動学を自前で参照したいので yourdfpy のモデルを保持しておく
    # (ViserUrdf.update_cfg はこのオブジェクトをそのまま更新してくれる)。
    arm_model = load_urdf(arm_urdf_path)
    arm_urdf = viser.extras.ViserUrdf(server, urdf_or_path=arm_model)

    # ハンドは別URDFなので中継フレームを親にしてぶら下げる。
    # ViserUrdf が内部で使うノード名の規則はviserのバージョンで変わるため、
    # そこには依存せず自分で作ったフレームを親にする。
    tool_frame = server.scene.add_frame(mount_node_name, show_axes=False)
    hand_urdf = viser.extras.ViserUrdf(
        server, urdf_or_path=load_urdf(hand_urdf_path), root_node_name=mount_node_name
    )
    # ハンド側に可動関節はない想定だが、固定ジョイントの姿勢を一度反映しておく。
    hand_urdf.update_cfg(np.zeros(len(hand_urdf.get_actuated_joint_names())))

    # 全関節0の姿勢で表示し、中継フレームをそのときの手先姿勢に合わせる。
    arm_urdf.update_cfg(np.zeros(len(arm_urdf.get_actuated_joint_names())))
    T_world_mount = arm_model.get_transform(mount_link)
    tool_frame.wxyz = vtf.SO3.from_matrix(T_world_mount[:3, :3]).wxyz
    tool_frame.position = T_world_mount[:3, 3]

    return arm_urdf, hand_urdf, tool_frame


def main(
    arm_urdf_path: Path = DEFAULT_ARM_URDF,
    hand_urdf_path: Path = DEFAULT_HAND_URDF,
    mount_link: str = DEFAULT_MOUNT_LINK,
) -> None:
    """アーム+ハンドを表示するViserサーバーを起動し、Ctrl+Cまで動かし続ける。"""
    server = viser.ViserServer()
    server.scene.add_grid("/grid", width=2.0, height=2.0)
    add_arm_with_hand(server, arm_urdf_path, hand_urdf_path, mount_link=mount_link)

    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        print("Stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm", type=Path, default=DEFAULT_ARM_URDF, help="アームのURDFファイル"
    )
    parser.add_argument(
        "--hand", type=Path, default=DEFAULT_HAND_URDF, help="ハンドのURDFファイル"
    )
    parser.add_argument(
        "--mount-link",
        default=DEFAULT_MOUNT_LINK,
        help="ハンドを取り付けるアーム側のリンク名",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.arm, args.hand, mount_link=args.mount_link)
