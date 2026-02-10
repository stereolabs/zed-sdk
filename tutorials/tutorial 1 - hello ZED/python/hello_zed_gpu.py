########################################################################
#
# Copyright (c) 2025, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

""" HELLO ZED GPU TEST SCRIPT
Test script to verify CuPy integration with the ZED SDK Python API.

It retrieves an image from the ZED camera, processes it on the GPU using CuPy,
and runs various tests to validate GPU functionality.
"""

import numpy as np
import time
import pyzed.sl as sl

# Test CuPy availability
try:
    import cupy as cp  # type: ignore[import-not-found]
    print("✅ CuPy detected - GPU acceleration available")
    print(f"   CuPy version: {cp.__version__}")
    print(f"   CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
except ModuleNotFoundError:
    raise ModuleNotFoundError("⚠️  CuPy not available - please install CuPy for GPU acceleration.\n"
                      "   pip install cupy-cuda11x  # For CUDA 11.x\n"
                      "   pip install cupy-cuda12x  # For CUDA 12.x")


GRAYSCALE_GPU_WEIGHTS = cp.array([0.299, 0.587, 0.114], dtype=cp.float32)
GRAYSCALE_CPU_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def test_gpu_image_processing(sl_img: sl.Mat):
    """Test the GPU image processing function."""
    print("\n🧪 Testing GPU image processing (basic grayscale conversion)...")

    try:
        # Generate test data
        cp_array = sl_img.get_data(sl.MEM.GPU, deep_copy=False)
        assert cp_array is not None, "Failed to retrieve image data from ZED camera"
        assert isinstance(cp_array, cp.ndarray), "Image data is not a CuPy array"
        assert cp_array.shape == (sl_img.get_height(), sl_img.get_width(), sl_img.get_channels()), "Shape mismatch for image data"
        np_array = sl_img.get_data(sl.MEM.CPU, deep_copy=False)
        assert cp_array.shape == np_array.shape, "Failed to convert image to CPU format"
        print(f"   Input image: {cp_array.shape}")

        # Example processing: Convert to grayscale (simple operation)
        if sl_img.get_channels() == 4:  # RGBA
            gray_image = cp.dot(cp_array[..., :3], GRAYSCALE_GPU_WEIGHTS)
        elif sl_img.get_channels() == 3:  # RGB
            gray_image = cp.dot(cp_array, GRAYSCALE_GPU_WEIGHTS)
        else:
            raise ValueError("Unsupported image format, expected RGB or RGBA")

        print(f"   Processed image: {gray_image.shape}")
        assert gray_image.ndim == 2, "Processed image should be grayscale (2D array)"

        print("✅ GPU processing test passed!")

    except cp.cuda.memory.OutOfMemoryError:
        print("❌ GPU out of memory - reduce test data size")
        raise
    except Exception as e:
        print(f"❌ GPU processing test failed: {e}")
        raise


def test_memory_types():
    """Test different memory allocation strategies."""
    print("\n💾 Testing memory allocation strategies...")

    test_size = (480, 640, 4)

    # CPU allocation
    cpu_data = np.random.rand(*test_size).astype(np.float32)
    print(f"   CPU allocation: {cpu_data.shape}, {cpu_data.dtype}")

    # GPU allocation
    gpu_data = cp.random.rand(*test_size, dtype=cp.float32)
    print(f"   GPU allocation: {gpu_data.shape}, {gpu_data.dtype}")

    # Test CPU to GPU transfer
    gpu_from_cpu = cp.asarray(cpu_data)
    print(f"   CPU->GPU transfer: {gpu_from_cpu.shape}")

    # Test GPU to CPU transfer
    cpu_from_gpu = cp.asnumpy(gpu_data)
    print(f"   GPU->CPU transfer: {cpu_from_gpu.shape}")

    assert cpu_data.shape == test_size, "Shape mismatch for CPU data"
    assert gpu_data.shape == test_size, "Shape mismatch for GPU data"
    assert gpu_from_cpu.shape == test_size, "Shape mismatch for GPU from CPU data"
    assert cpu_from_gpu.shape == test_size, "Shape mismatch for CPU from GPU data"

    print("✅ Memory allocation test passed!")


def test_gpu_memory_usage():
    """Test GPU memory usage and cleanup."""
    print("\n🔍 Testing GPU memory usage...")

    # Check initial memory
    initial_memory = cp.get_default_memory_pool().used_bytes()
    print(f"   Initial GPU memory usage: {initial_memory / 1024**2:.1f} MB")

    # Allocate large array
    large_array = cp.random.rand(1000, 1000, 4, dtype=cp.float32)
    after_alloc_memory = cp.get_default_memory_pool().used_bytes()
    print(f"   After allocation: {after_alloc_memory / 1024**2:.1f} MB")
    assert after_alloc_memory > initial_memory, "Memory should increase after allocation"

    # Free memory
    del large_array
    cp.get_default_memory_pool().free_all_blocks()
    final_memory = cp.get_default_memory_pool().used_bytes()
    print(f"   After cleanup: {final_memory / 1024**2:.1f} MB")
    assert final_memory < after_alloc_memory, "Memory should decrease after cleanup"

    print("✅ GPU memory test passed!")


def test_data_integrity():
    """Test that GPU operations preserve data integrity."""
    print("\n🔬 Testing data integrity...")

    # Create known test data
    test_data = np.array([[[ 1.0,  2.0,  3.0, 1.0],
                           [ 4.0,  5.0,  6.0, 1.0]],
                          [[ 7.0,  8.0,  9.0, 1.0],
                           [10.0, 11.0, 12.0, 1.0]]], dtype=np.float32)

    # Process on GPU
    gpu_data = cp.asarray(test_data)

    # Simple operation: multiply by 2
    gpu_result = gpu_data * 2.0
    cpu_result = cp.asnumpy(gpu_result)

    # Expected result
    expected = test_data * 2.0

    # Verify results match
    assert np.allclose(cpu_result, expected), "GPU and expected results don't match"
    print(f"   Data integrity verified: {cpu_result.shape}")

    print("✅ Data integrity test passed!")


def benchmark_processing(sl_img: sl.Mat):
    """Simple benchmark comparing CPU vs GPU processing."""
    print("\n⚡ Running performance benchmark...")

    print(f"   Benchmark image size: {sl_img.get_width()}x{sl_img.get_height()}")

    # CPU benchmark
    start_time = time.time()
    for _ in range(10):
        np_array = sl_img.get_data(sl.MEM.CPU, deep_copy=False)
        gray_image_cpu = np.dot(np_array[..., :3], GRAYSCALE_CPU_WEIGHTS)
    cpu_time = time.time() - start_time
    print(f"   CPU processing (10 iterations): {cpu_time * 1000:.3f} milliseconds")

    # GPU benchmark
    start_time = time.time()
    for _ in range(10):
        gpu_array = sl_img.get_data(sl.MEM.GPU, deep_copy=False)
        gray_image_gpu = cp.dot(gpu_array[..., :3], GRAYSCALE_GPU_WEIGHTS)
        cp.cuda.Stream.null.synchronize()  # Wait for GPU operations to complete
    gpu_time = time.time() - start_time

    print(f"   GPU processing (10 iterations): {gpu_time * 1000:.3f} milliseconds")
    print(f"   Speedup: {cpu_time/gpu_time:.1f}x")

    if gpu_time < cpu_time:
        print("🚀 GPU processing is faster!")
    else:
        print("💻 CPU processing is faster (small dataset or GPU overhead)")


if __name__ == "__main__":
    print("ZED SDK CuPy Integration Test")
    print("=" * 40)

    # Create a Camera object
    zed = sl.Camera()

    # Open the camera
    print("Opening ZED camera...")
    init_params = sl.InitParameters()
    init_params.sdk_verbose = 1
    err = zed.open(init_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print("❌ Failed to open ZED camera. Ensure it is connected and accessible.")
        exit(1)
    print("ZED camera opened successfully.")

    # Try 50 grabs to test point cloud retrieval
    sl_img = sl.Mat()
    retrieved = False
    i = 0
    runtime_parameters = sl.RuntimeParameters()
    print("Retrieving image data...")
    while i < 50:
        if zed.grab(runtime_parameters) <= sl.ERROR_CODE.SUCCESS:
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_image(sl_img, sl.VIEW.LEFT, sl.MEM.GPU)
            retrieved = True
            break
        i += 1
    if not retrieved:
        print("❌ Failed to retrieve image data after 50 attempts.")
        zed.close()
        exit(1)
    # Update the image from GPU to CPU
    sl_img.update_cpu_from_gpu()
    print(f"Retrieved image on GPU: {sl_img.get_width()}x{sl_img.get_height()}")

    try:
        test_gpu_image_processing(sl_img)
        print("=" * 40)
        test_memory_types()
        print("=" * 40)
        test_gpu_memory_usage()
        print("=" * 40)
        test_data_integrity()
        print("=" * 40)
        benchmark_processing(sl_img)
        print("=" * 40)

        print("\n🎉 All tests completed!")
        print("   Your system is ready for GPU-accelerated ZED processing with the Python API!")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        # Close the camera
        zed.close()
        exit(1)

    # Close the camera
    zed.close()
