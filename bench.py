import glob, time, yaml, torch
from aidet.data import build_dataloaders
from aidet.models import build_model


def main():
    cfg = yaml.safe_load(open(sorted(glob.glob("configs/lr_sweep/*resnet50*lr1e-4*.yaml"))[0]))
    dev = torch.device("mps")

    # 1. Data loading only
    import random
    train_loader, _ = build_dataloaders(cfg, dev)
    ds = train_loader.dataset
    idx = random.sample(range(len(ds)), 40)
    t = time.time()
    for i in idx:
        ds[i]
    per_img = (time.time() - t) / 40
    batch_s = per_img * 64 / cfg["num_workers"]
    print(f"data:  {per_img*1000:.0f} ms/image -> {batch_s:.2f} s/batch -> {batch_s*187/60:.1f} min/epoch")
    # train_loader, _ = build_dataloaders(cfg, dev)
    # it = iter(train_loader); next(it)                 # warm up workers
    # t = time.time()
    # for _ in range(10):
    #     next(it)
    # data_s = (time.time() - t) / 10
    # print(f"data:  {data_s:.2f} s/batch -> {data_s*187/60:.1f} min/epoch if data-bound")

    # 2. Model only, random input
    m = build_model("resnet50", "scratch").to(dev)
    opt = torch.optim.AdamW(m.parameters())
    x = torch.randn(64, 3, 224, 224, device=dev)
    y = torch.zeros(64, device=dev)

    def step():
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x).reshape(-1), y)
        opt.zero_grad(); loss.backward(); opt.step()

    for _ in range(3):
        step()
    torch.mps.synchronize(); t = time.time()
    for _ in range(5):
        step()
    torch.mps.synchronize()
    gpu_s = (time.time() - t) / 5
    print(f"model: {gpu_s:.2f} s/step  -> {gpu_s*187/60:.1f} min/epoch if GPU-bound")


if __name__ == "__main__":
    main()