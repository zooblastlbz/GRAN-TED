import subprocess
import os

def clear_gpu_memory():
    try:
        # 获取所有 GPU 上的进程信息
        result = subprocess.check_output("nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader", shell=True)
        lines = result.decode().strip().split('\n')

        if not lines or lines == ['']:
            print("✅ 当前无显存占用")
            return

        killed = []
        for line in lines:
            pid = line.split(',')[0].strip()
            try:
                os.kill(int(pid), 9)  # 强制杀死进程
                killed.append(pid)
            except Exception as e:
                print(f"❌ 无法杀死进程 {pid}：{e}")

        if killed:
            print(f"✅ 已清除以下占用 GPU 的进程：{', '.join(killed)}")
        else:
            print("⚠️ 未能杀死任何进程，可能无权限或显存已释放")

    except subprocess.CalledProcessError as e:
        print("❌ 执行 nvidia-smi 失败：", e)

if __name__ == "__main__":
    clear_gpu_memory()
