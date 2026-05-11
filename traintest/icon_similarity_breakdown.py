import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _is_image_file(p: str) -> bool:
    ext = Path(p).suffix.lower()
    return ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def _resolve_icon_path(p: str, mode: str) -> Tuple[str, str]:
    p = str(p)
    if mode not in {"auto", "image", "project"}:
        raise ValueError(f"Unknown mode: {mode}")

    if mode in {"auto", "image"} and _is_image_file(p):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p, "image"

    if mode in {"auto", "project"}:
        from traintest.icon_detection import find_icon_path

        icon = find_icon_path(p)
        if icon is None:
            raise FileNotFoundError(f"未找到图标: {p}")
        return icon, "project"

    raise FileNotFoundError(p)


def _load_and_preprocess_image(image_path: str, device):
    from PIL import Image
    import torchvision.transforms as transforms
    from traintest.icon_detection import convert_rgba_to_rgb

    image = Image.open(image_path)
    if image.mode != "RGB":
        image = convert_rgba_to_rgb(image)
    image = image.resize((224, 224))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    x = transform(image).unsqueeze(0).to(device)
    return x


def _extract_content_style(model, x, device):
    import torch

    model.eval()
    with torch.no_grad():
        feats = model(x)

        f_content = feats["fc2"]
        f_content = torch.nn.functional.normalize(f_content, p=2, dim=1)

        f_style = feats["28"]
        b, ch, h, w = f_style.size()
        f_style = f_style.view(b, ch, h * w)
        grams = torch.bmm(f_style, f_style.transpose(1, 2)) / (ch * h * w)
        grams = grams.view(b, -1)
        grams = torch.nn.functional.normalize(grams, p=2, dim=1)

        return f_content.to(device), grams.to(device)


def compute_icon_similarities(p1: str, p2: str, alpha: float, mode: str) -> dict:
    import torch
    from traintest.icon_detection import VGG19FeatureExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_names = ["28", "fc2"]
    model = VGG19FeatureExtractor(layer_names).to(device).eval()

    icon1, p1_kind = _resolve_icon_path(p1, mode=mode)
    icon2, p2_kind = _resolve_icon_path(p2, mode=mode)

    x1 = _load_and_preprocess_image(icon1, device=device)
    x2 = _load_and_preprocess_image(icon2, device=device)

    c1, s1 = _extract_content_style(model, x1, device=device)
    c2, s2 = _extract_content_style(model, x2, device=device)

    content_sim = float((c1 * c2).sum(dim=1).item())
    style_sim = float((s1 * s2).sum(dim=1).item())
    overall_sim = float(alpha * content_sim + (1.0 - alpha) * style_sim)

    return {
        "device": str(device),
        "alpha": alpha,
        "p1": p1,
        "p2": p2,
        "icon1_path": icon1,
        "icon2_path": icon2,
        "p1_kind": p1_kind,
        "p2_kind": p2_kind,
        "content_similarity": content_sim,
        "style_similarity": style_sim,
        "overall_similarity": overall_sim,
    }


def main():
    parser = argparse.ArgumentParser(description="输出 Icon 内容/风格/加权相似度得分")

    parser.add_argument("--p1", help="图标图片路径，或 APK 项目目录（包含 images/）")
    parser.add_argument("--p2", help="图标图片路径，或 APK 项目目录（包含 images/）")
    parser.add_argument("--mode", choices=["auto", "image", "project"], default="auto", help="输入解析模式")
    parser.add_argument("--alpha", type=float, default=0.6, help="内容相似度权重 alpha（overall=alpha*content+(1-alpha)*style）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")

    parser.add_argument("--root", default=None, help="AndroZoo根目录（启用批量检测模式）")
    parser.add_argument("--sample-fraction", type=float, default=0.25, help="抽样的组比例")
    parser.add_argument("--threshold", type=float, default=0.6, help="判定为相似的阈值")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader线程数")
    parser.add_argument("--batch-size", type=int, default=64, help="特征提取批大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--intermediate-root", default="/newdisk/liuzhuowu/baseline/suidroid", help="中间文件根目录")
    parser.add_argument("--result-root", default="/newdisk/liuzhuowu/baseline/suidroid_result", help="结果根目录")
    args = parser.parse_args()

    if args.root is not None:
        from traintest.icon_detection import run_icon_detection

        metrics = run_icon_detection(args)
        result_path = os.path.join(args.result_root, "image", "image_all_groups_sampled.xlsx")
        if metrics is not None:
            print(json.dumps({"metrics": metrics, "result_xlsx": result_path}, ensure_ascii=False, indent=2))
        else:
            print(json.dumps({"metrics": None, "result_xlsx": result_path}, ensure_ascii=False, indent=2))
        return

    if not args.p1 or not args.p2:
        raise ValueError("需要提供 --p1 与 --p2，或提供 --root 启用批量检测模式")

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha 需要在 [0,1] 范围内")

    out = compute_icon_similarities(args.p1, args.p2, alpha=float(args.alpha), mode=args.mode)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(f"device: {out['device']}")
    print(f"p1_kind: {out['p1_kind']} | icon1: {out['icon1_path']}")
    print(f"p2_kind: {out['p2_kind']} | icon2: {out['icon2_path']}")
    print(f"content_similarity: {out['content_similarity']:.6f}")
    print(f"style_similarity:   {out['style_similarity']:.6f}")
    print(f"alpha: {out['alpha']:.3f}")
    print(f"overall_similarity: {out['overall_similarity']:.6f}")


if __name__ == "__main__":
    main()
