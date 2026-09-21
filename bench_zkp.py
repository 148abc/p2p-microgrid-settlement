""""""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

os.chdir(os.path.join(os.path.dirname(__file__), "..", "zkp_circuit"))

import subprocess

def run_bench(circuit="wp3"):
    """"""
    print(f"\n=== {circuit} 电路 ===")

    result = subprocess.run(["go", "run", ".", "setup"], capture_output=True, text=True)
    if result.returncode != 0:
        print("setup 失败:", result.stderr[-200:])
        return None

    for line in result.stdout.split('\n'):
        if 'nbConstraints' in line:
            print(line.strip())

    result = subprocess.run(["go", "run", ".", "test"], capture_output=True, text=True)
    if result.returncode != 0:
        print("test 失败:", result.stderr[-500:])
        return None

    sizes = {}
    for fname in ["proof.json", "vk.json", "pk.bin"]:
        if os.path.exists(fname):
            sizes[fname] = os.path.getsize(fname)
            print(f"  {fname}: {sizes[fname]} bytes")

    return sizes

if __name__ == "__main__":
    print("ZKP 性能测试")
    print("=" * 50)

    wp3_sizes = run_bench("wp3")

    wp4_sizes = run_bench("wp4_eta1")

    print("\n" + "=" * 50)
    print("汇总：")
    print(f"WP3 (24 constraints):  proof={wp3_sizes.get('proof.json','-')}B, vk={wp3_sizes.get('vk.json','-')}B")
    print(f"WP4_eta1 (105 constraints): proof={wp4_sizes.get('proof.json','-')}B, vk={wp4_sizes.get('vk.json','-')}B")
