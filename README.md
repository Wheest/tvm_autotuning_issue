# tvm autotuning issue

Run autotuning with flags for example:

```
python tvm_autotune.py --target_type remote --device_key hikey --target_string 'llvm -device=arm_cpu -target=aarch64-linux-gnu' --output_path /tmp/ --host_port 9190  --trials 5

python tvm_autotune.py --target_type local  --target_string 'llvm -target=x86_64-linux-gnu -mcpu=core-avx2' --output_path /tmp/ --trials 5 --model resnet34
```

Models are included.
