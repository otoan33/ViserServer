import time
import viser

def main():
    # 描画サーバー立ち上げ
    server = viser.ViserServer()

    # 球を配置
    server.scene.add_icosphere(
        name="/hello_sphere",
        radius=0.5,
        color=(255, 0, 0),  # Red
        position=(1.0, 0.0, 0.0),
    )
    
    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to exit")

    # 無限ループ
    try:
        while True:
            time.sleep(10.0)
    # Ctrl+C で Exit
    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    main()