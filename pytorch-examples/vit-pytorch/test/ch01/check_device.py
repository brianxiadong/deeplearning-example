#!/usr/bin/env python3
"""
设备检测脚本
检查当前系统的 PyTorch 设备支持情况
"""

import torch
import platform
import sys

def check_pytorch_info():
    """检查 PyTorch 基本信息"""
    print("🔍 PyTorch 环境信息")
    print("=" * 40)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"Python 版本: {sys.version}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    
    if platform.system() == "Darwin":
        # macOS 系统，显示芯片信息
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                cpu_info = result.stdout.strip()
                print(f"处理器: {cpu_info}")
        except:
            pass
    
    print()

def check_cuda_support():
    """检查 CUDA 支持"""
    print("🚀 CUDA 支持检查")
    print("=" * 40)
    
    if torch.cuda.is_available():
        print("✅ CUDA 可用")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # 测试 CUDA 张量操作
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            print("✅ CUDA 张量操作测试通过")
        except Exception as e:
            print(f"❌ CUDA 张量操作测试失败: {e}")
    else:
        print("❌ CUDA 不可用")
        if platform.system() == "Windows":
            print("💡 Windows 用户可以安装 CUDA 版本的 PyTorch:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        elif platform.system() == "Linux":
            print("💡 Linux 用户可以安装 CUDA 版本的 PyTorch:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print()

def check_mps_support():
    """检查 Apple MPS 支持"""
    print("🍎 Apple MPS 支持检查")
    print("=" * 40)
    
    if platform.system() != "Darwin":
        print("❌ MPS 仅在 macOS 系统上可用")
        print()
        return
    
    if hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available():
            print("✅ MPS 可用")
            print("✅ Apple Silicon 加速已启用")
            
            # 测试 MPS 张量操作
            try:
                x = torch.randn(100, 100).to('mps')
                y = torch.randn(100, 100).to('mps')
                z = torch.mm(x, y)
                print("✅ MPS 张量操作测试通过")
                
                # 性能简单测试
                import time
                
                # CPU 测试
                x_cpu = torch.randn(1000, 1000)
                y_cpu = torch.randn(1000, 1000)
                start_time = time.time()
                for _ in range(10):
                    z_cpu = torch.mm(x_cpu, y_cpu)
                cpu_time = time.time() - start_time
                
                # MPS 测试
                x_mps = torch.randn(1000, 1000).to('mps')
                y_mps = torch.randn(1000, 1000).to('mps')
                start_time = time.time()
                for _ in range(10):
                    z_mps = torch.mm(x_mps, y_mps)
                    torch.mps.synchronize()  # 确保操作完成
                mps_time = time.time() - start_time
                
                speedup = cpu_time / mps_time
                print(f"🚀 MPS 相对 CPU 加速比: {speedup:.1f}x")
                
            except Exception as e:
                print(f"❌ MPS 张量操作测试失败: {e}")
                print("💡 尝试更新 PyTorch: pip install --upgrade torch torchvision")
        else:
            print("❌ MPS 不可用")
            print("💡 可能的原因:")
            print("   - 不是 Apple Silicon (M1/M2/M3) 芯片")
            print("   - PyTorch 版本过旧")
            print("   - macOS 版本过旧 (需要 macOS 12.3+)")
    else:
        print("❌ 当前 PyTorch 版本不支持 MPS")
        print("💡 请更新到支持 MPS 的 PyTorch 版本:")
        print("   pip install --upgrade torch torchvision")
    
    print()

def check_cpu_info():
    """检查 CPU 信息"""
    print("💻 CPU 信息")
    print("=" * 40)
    
    # 测试 CPU 性能
    import time
    
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    
    start_time = time.time()
    for _ in range(10):
        z = torch.mm(x, y)
    cpu_time = time.time() - start_time
    
    print(f"CPU 矩阵乘法性能: {cpu_time:.3f}s (10次 1000x1000)")
    print(f"CPU 线程数: {torch.get_num_threads()}")
    print("✅ CPU 可用")
    print()

def get_recommended_device():
    """获取推荐的设备"""
    print("🎯 推荐设备配置")
    print("=" * 40)
    
    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 推荐使用 CUDA GPU 进行训练")
        print("   - 最快的训练速度")
        print("   - 支持大批次训练")
        print("   - 适合长时间训练")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("🍎 推荐使用 Apple MPS 进行训练")
        print("   - 比 CPU 快 3-5 倍")
        print("   - 低功耗和发热")
        print("   - 统一内存架构")
    else:
        device = "cpu"
        print("💻 使用 CPU 进行训练")
        print("   - 兼容性最好")
        print("   - 训练速度较慢")
        print("   - 适合小模型和调试")
    
    print(f"\n推荐设备: {device}")
    print()

def main():
    """主函数"""
    print("🔧 PyTorch 设备支持检测工具")
    print("=" * 50)
    print()
    
    # 检查各种设备支持
    check_pytorch_info()
    check_cuda_support()
    check_mps_support()
    check_cpu_info()
    get_recommended_device()
    
    print("🎉 检测完成！")
    print()
    print("💡 使用建议:")
    print("   - 训练大模型: 优先选择 CUDA GPU")
    print("   - Apple 设备: 使用 MPS 加速")
    print("   - 调试和小实验: CPU 即可")
    print("   - 生产环境: 建议使用 GPU 加速")

if __name__ == "__main__":
    main()
