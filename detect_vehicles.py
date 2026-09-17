#!/usr/bin/env python3

import csv
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
except Exception:
    torch = None


VEHICLE_CLASSES = {
    2: 'Легковой автомобиль',
    3: 'Мотоцикл',
    5: 'Автобус',
    7: 'Грузовой автомобиль',
}

KNOWN_MODELS = {
    'KAMAZ': {'manufacturer': 'КАМАЗ', 'models': ['КАМАЗ-55111', 'КАМАЗ-55102', 'КАМАЗ-53215', 'КАМАЗ-65115']},
    'КАМАЗ': {'manufacturer': 'КАМАЗ', 'models': ['КАМАЗ-55111', 'КАМАЗ-55102', 'КАМАЗ-53215', 'КАМАЗ-65115']},
    'КИРОВЕЦ': {'manufacturer': 'Кировец (ПТЗ)', 'models': ['К-700А', 'К-744']},
    'LOVOL': {'manufacturer': 'Lovol (Foton Lovol)', 'models': ['Lovol TD1304']},
    'BELARUS': {'manufacturer': 'МТЗ', 'models': ['Беларус МТЗ-82']},
    'БЕЛАРУС': {'manufacturer': 'МТЗ', 'models': ['Беларус МТЗ-82']},
    'МТЗ': {'manufacturer': 'МТЗ', 'models': ['Беларус МТЗ-82']},
    'ГАЗ': {'manufacturer': 'ГАЗ', 'models': ['ГАЗ-53', 'ГАЗ-3307']},
}

PLATE_ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789 '


def setup_models():
    """Загружает YOLOv8 и EasyOCR."""
    vehicle_model = YOLO('yolov8x.pt')
    use_gpu = bool(torch and torch.cuda.is_available())
    reader = easyocr.Reader(['en', 'ru'], gpu=use_gpu)
    return vehicle_model, reader


def detect_vehicles(model, image_path: str):
    results = model(image_path, conf=0.25, classes=[2, 3, 5, 7])
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                'class_id': cls_id,
                'class_name': VEHICLE_CLASSES.get(cls_id, 'Unknown'),
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],
            })

    return detections


def clamp_bbox(bbox: List[int], image_shape) -> Optional[List[int]]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return [x1, y1, x2, y2]


def crop_by_bbox(image, bbox: List[int]):
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def rel_bbox(parent_bbox: List[int], rx1: float, ry1: float, rx2: float, ry2: float, image_shape) -> Optional[List[int]]:
    px1, py1, px2, py2 = parent_bbox
    pw = px2 - px1
    ph = py2 - py1
    bbox = [
        px1 + int(pw * rx1),
        py1 + int(ph * ry1),
        px1 + int(pw * rx2),
        py1 + int(ph * ry2),
    ]
    return clamp_bbox(bbox, image_shape)


def iou(box_a: List[int], box_b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)


def deduplicate_candidates(candidates: List[Dict], threshold: float = 0.6) -> List[Dict]:
    result = []
    for cand in sorted(candidates, key=lambda x: x['score'], reverse=True):
        if all(iou(cand['bbox'], kept['bbox']) < threshold for kept in result):
            result.append(cand)
    return result


def build_plate_search_regions(image, vehicle_bbox: List[int], vehicle_class_name: str = '') -> List[Dict]:
    """
    Номер обычно находится:
    - у грузовиков: нижний центр / нижний левый-правый угол кузова / зона бампера;
    - у тракторов: передняя центральная зона или задний нижний центр.

    Вместо поиска по всей нижней половине проверяем несколько прицельных зон.
    """
    x1, y1, x2, y2 = vehicle_bbox
    vw = x2 - x1
    vh = y2 - y1
    aspect = vw / max(vh, 1)

    if aspect < 1.2:
        templates = [
            ('tractor_front_center', 0.22, 0.30, 0.78, 0.68, 0.95),
            ('tractor_lower_center', 0.22, 0.58, 0.78, 0.90, 1.00),
            ('tractor_lower_left', 0.00, 0.52, 0.55, 0.92, 0.78),
            ('tractor_lower_right', 0.45, 0.52, 1.00, 0.92, 0.78),
        ]
    else:
        templates = [
            ('rear_lower_center', 0.18, 0.56, 0.82, 0.92, 1.00),
            ('rear_lower_narrow', 0.28, 0.62, 0.72, 0.90, 1.00),
            ('front_mid_center', 0.18, 0.34, 0.82, 0.74, 0.92),
            ('lower_left', 0.00, 0.54, 0.52, 0.92, 0.78),
            ('lower_right', 0.48, 0.54, 1.00, 0.92, 0.78),
        ]

    regions = []
    for name, rx1, ry1, rx2, ry2, weight in templates:
        bbox = rel_bbox(vehicle_bbox, rx1, ry1, rx2, ry2, image.shape)
        if bbox:
            regions.append({'name': name, 'bbox': bbox, 'weight': weight})
    return regions


def preprocess_roi(gray: np.ndarray) -> List[np.ndarray]:
    """Готовит несколько бинарных масок, чтобы не зависеть от одной эвристики."""
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    blackhat = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, rect_kernel)
    grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = np.absolute(grad_x)
    grad_x = (255 * ((grad_x - grad_x.min()) / (grad_x.max() - grad_x.min() + 1e-6))).astype('uint8')
    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
    _, grad_thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, sq_kernel)

    canny = cv2.Canny(clahe, 70, 180)
    canny = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)

    return [grad_thresh, adaptive, canny]


def score_candidate(region_gray: np.ndarray, local_bbox: Tuple[int, int, int, int], roi_weight: float, vehicle_area: int) -> float:
    x, y, w, h = local_bbox
    crop = region_gray[y:y + h, x:x + w]
    if crop.size == 0:
        return 0.0

    ratio = w / max(h, 1)
    area_ratio = (w * h) / max(vehicle_area, 1)

    ratio_score = max(0.0, 1.0 - abs(ratio - 3.8) / 3.8)
    area_score = 1.0 if 0.004 <= area_ratio <= 0.06 else 0.45 if 0.002 <= area_ratio <= 0.12 else 0.0

    edges = cv2.Canny(crop, 70, 180)
    edge_density = float(np.mean(edges > 0))
    edge_score = min(1.0, edge_density * 4.5)

    bright_ratio = float(np.mean(crop > 145))
    dark_ratio = float(np.mean(crop < 110))
    contrast_score = min(1.0, (bright_ratio + dark_ratio) * 1.2)

    center_x = (x + w / 2) / region_gray.shape[1]
    center_y = (y + h / 2) / region_gray.shape[0]
    center_score = max(0.0, 1.0 - abs(center_x - 0.5) * 1.5)
    lower_score = min(1.0, 0.45 + center_y)

    score = (
        roi_weight * 0.18 +
        ratio_score * 0.18 +
        area_score * 0.14 +
        edge_score * 0.18 +
        contrast_score * 0.14 +
        center_score * 0.08 +
        lower_score * 0.10
    )
    return float(score)


def extract_plate_candidates_from_roi(image, vehicle_bbox: List[int], roi: Dict) -> List[Dict]:
    roi_bbox = roi['bbox']
    roi_crop = crop_by_bbox(image, roi_bbox)
    if roi_crop.size == 0:
        return []

    roi_gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
    masks = preprocess_roi(roi_gray)

    vx1, vy1, vx2, vy2 = vehicle_bbox
    vehicle_area = max(1, (vx2 - vx1) * (vy2 - vy1))
    min_w = max(28, int((vx2 - vx1) * 0.07))
    min_h = max(10, int((vy2 - vy1) * 0.025))

    candidates = []
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / max(h, 1)
            area_ratio = (w * h) / vehicle_area

            if w < min_w or h < min_h:
                continue
            if not (2.0 <= ratio <= 7.8):
                continue
            if not (0.002 <= area_ratio <= 0.14):
                continue

            global_bbox = [roi_bbox[0] + x, roi_bbox[1] + y, roi_bbox[0] + x + w, roi_bbox[1] + y + h]
            score = score_candidate(roi_gray, (x, y, w, h), roi['weight'], vehicle_area)
            candidates.append({
                'bbox': global_bbox,
                'score': score,
                'source': roi['name'],
            })

    return candidates


def heuristic_fallback_candidates(image, vehicle_bbox: List[int]) -> List[Dict]:
    """Если контуры не нашли ничего полезного, всё равно проверяем несколько фиксированных коробок."""
    fallbacks = []
    specs = [
        ('fallback_center_low', 0.30, 0.66, 0.70, 0.86, 0.62),
        ('fallback_center_mid', 0.28, 0.48, 0.72, 0.68, 0.56),
        ('fallback_left_low', 0.08, 0.62, 0.48, 0.86, 0.46),
        ('fallback_right_low', 0.52, 0.62, 0.92, 0.86, 0.46),
    ]
    for name, rx1, ry1, rx2, ry2, score in specs:
        bbox = rel_bbox(vehicle_bbox, rx1, ry1, rx2, ry2, image.shape)
        if bbox:
            fallbacks.append({'bbox': bbox, 'score': score, 'source': name})
    return fallbacks


def normalize_plate_text(text: str) -> str:
    text = text.upper().replace('|', '1')
    text = re.sub(r'[^A-ZА-Я0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def plate_format_score(text: Optional[str]) -> float:
    if not text:
        return 0.0

    text = normalize_plate_text(text)
    compact = text.replace(' ', '')
    if len(compact) < 4 or len(compact) > 10:
        return 0.05

    score = 0.15
    has_digits = any(ch.isdigit() for ch in compact)
    has_letters = any(ch.isalpha() for ch in compact)
    if has_digits:
        score += 0.20
    if has_letters:
        score += 0.20
    if has_digits and has_letters:
        score += 0.10

    patterns = [
        r'^\d{3}[A-ZА-Я]{2,3}\d{2}$',
        r'^[A-ZА-Я]{2,3}\d{4,5}$',
        r'^[A-ZА-Я]{2,3}\d{1,3}[A-ZА-Я]{0,2}\d{2}$',
        r'^\d{1,4}[A-ZА-Я]{2,3}\d{2}$',
    ]
    if any(re.fullmatch(p, compact) for p in patterns):
        score += 0.30

    if compact.endswith('10'):
        score += 0.10
    if compact[:3].isdigit():
        score += 0.05

    return min(1.0, score)


def run_ocr_variants(reader, plate_crop) -> Tuple[Optional[str], float]:
    if plate_crop is None or plate_crop.size == 0:
        return None, 0.0

    plate_crop = cv2.resize(plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 25, 25)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv = 255 - otsu

    variants = [gray, clahe, otsu, otsu_inv]

    best_text = None
    best_score = 0.0

    for variant in variants:
        results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            allowlist=PLATE_ALLOWLIST,
        )
        if not results:
            continue

        raw_text = ' '.join(item[1] for item in results)
        text = normalize_plate_text(raw_text)
        conf = float(sum(item[2] for item in results) / len(results))
        format_score = plate_format_score(text)
        score = conf * 0.65 + format_score * 0.35

        if score > best_score:
            best_score = score
            best_text = text

    return best_text, best_score


def find_best_plate(reader, image, vehicle_bbox: List[int], vehicle_class_name: str = '') -> Optional[Dict]:
    """Ищет номер по нескольким ROI и выбирает лучший кандидат после OCR-проверки."""
    regions = build_plate_search_regions(image, vehicle_bbox, vehicle_class_name)
    candidates = []
    for roi in regions:
        candidates.extend(extract_plate_candidates_from_roi(image, vehicle_bbox, roi))

    candidates = deduplicate_candidates(candidates)
    if len(candidates) < 4:
        candidates.extend(heuristic_fallback_candidates(image, vehicle_bbox))
        candidates = deduplicate_candidates(candidates)

    best = None
    for cand in sorted(candidates, key=lambda x: x['score'], reverse=True)[:12]:
        plate_crop = crop_by_bbox(image, cand['bbox'])
        text, ocr_score = run_ocr_variants(reader, plate_crop)
        text_score = plate_format_score(text)
        final_score = cand['score'] * 0.45 + ocr_score * 0.35 + text_score * 0.20

        current = {
            'bbox': cand['bbox'],
            'source': cand['source'],
            'candidate_score': round(cand['score'], 4),
            'ocr_score': round(ocr_score, 4),
            'text_score': round(text_score, 4),
            'final_score': round(final_score, 4),
            'text': text,
        }

        if best is None or current['final_score'] > best['final_score']:
            best = current

    if best and best['final_score'] >= 0.34:
        return best
    return best if best and best['candidate_score'] >= 0.55 else None


def identify_manufacturer(image, vehicle_bbox, reader):
    x1, y1, x2, y2 = vehicle_bbox
    vehicle_crop = image[y1:y2, x1:x2]
    if vehicle_crop.size == 0:
        return 'Не определён', 'Не определена', 0.0

    results = reader.readtext(vehicle_crop, detail=1)
    for _, text, conf in results:
        text_upper = text.upper().strip()
        for key, info in KNOWN_MODELS.items():
            if key.upper() in text_upper:
                return info['manufacturer'], info['models'][0], conf

    return 'Не определён', 'Не определена', 0.0


def draw_detections(image_path, detections, output_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
        font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype('arial.ttf', 16)
            font_small = ImageFont.truetype('arial.ttf', 12)
        except (IOError, OSError):
            font = ImageFont.load_default()
            font_small = font

    for det in detections:
        bbox = det['vehicle_bbox']
        draw.rectangle(bbox, outline='#22d3ee', width=3)

        label = f"{det['manufacturer']} {det['model']} ({det['confidence']:.0%})"
        tx1, ty1, tx2, ty2 = draw.textbbox((bbox[0], max(0, bbox[1] - 22)), label, font=font)
        draw.rectangle([tx1 - 2, ty1 - 2, tx2 + 2, ty2 + 2], fill='black')
        draw.text((bbox[0], max(0, bbox[1] - 22)), label, fill='#22d3ee', font=font)

        if det.get('plate_bbox'):
            pbbox = det['plate_bbox']
            draw.rectangle(pbbox, outline='#fbbf24', width=2)
            plate_label = det.get('plate_text') or det.get('plate_source', 'plate')
            px1, py1, px2, py2 = draw.textbbox((pbbox[0], max(0, pbbox[1] - 16)), plate_label, font=font_small)
            draw.rectangle([px1 - 2, py1 - 2, px2 + 2, py2 + 2], fill='black')
            draw.text((pbbox[0], max(0, pbbox[1] - 16)), plate_label, fill='#fbbf24', font=font_small)

    img.save(output_path)
    print(f'  Сохранено: {output_path}')


def process_images(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'annotated').mkdir(exist_ok=True)

   

    print('\n[1/4] Загрузка моделей...')
    vehicle_model, reader = setup_models()
    print('  ✓ YOLOv8 загружен')
    print('  ✓ EasyOCR загружен')

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in image_extensions])
    print(f'\n[2/4] Найдено изображений: {len(image_files)}')

    print('\n[3/4] Обработка изображений...')
    all_results = []

    for idx, img_file in enumerate(image_files, 1):
        print(f'\n  --- Изображение {idx}/{len(image_files)}: {img_file.name} ---')
        image = cv2.imread(str(img_file))
        if image is None:
            print(f'  ⚠ Не удалось загрузить: {img_file.name}')
            continue

        vehicles = detect_vehicles(vehicle_model, str(img_file))
        print(f'  Обнаружено ТС: {len(vehicles)}')

        file_detections = []
        for v_idx, vehicle in enumerate(vehicles):
            manufacturer, model, brand_conf = identify_manufacturer(image, vehicle['bbox'], reader)
            plate_result = find_best_plate(reader, image, vehicle['bbox'], vehicle['class_name'])

            plate_text = plate_result['text'] if plate_result else None
            plate_bbox = plate_result['bbox'] if plate_result else None
            plate_final = plate_result['final_score'] if plate_result else 0.0
            plate_source = plate_result['source'] if plate_result else None

            result = {
                'id': len(all_results) + 1,
                'filename': img_file.name,
                'timestamp': datetime.now().isoformat(),
                'vehicle_type': vehicle['class_name'],
                'manufacturer': manufacturer,
                'model': model,
                'license_plate': plate_text or 'не читается',
                'plate_readable': bool(plate_text and plate_format_score(plate_text) >= 0.45),
                'plate_confidence': round(plate_final, 2),
                'detection_confidence': round(vehicle['confidence'], 2),
                'vehicle_bbox': vehicle['bbox'],
                'plate_bbox': plate_bbox,
                'plate_source': plate_source,
            }
            all_results.append(result)

            file_detections.append({
                'vehicle_bbox': vehicle['bbox'],
                'plate_bbox': plate_bbox,
                'plate_text': plate_text,
                'plate_source': plate_source,
                'manufacturer': manufacturer,
                'model': model,
                'confidence': vehicle['confidence'],
            })

            print(f"    ТС #{v_idx + 1}: {manufacturer} {model}")
            print(f"    Тип: {vehicle['class_name']}")
            print(f"    Номер: {plate_text or 'не читается'}")
            print(f"    Plate ROI: {plate_source or 'not-found'}")
            print(f"    Уверенность ТС: {vehicle['confidence']:.2f}, номер: {plate_final:.2f}")

        if file_detections:
            annotated_path = output_path / 'annotated' / f'annotated_{img_file.name}'
            draw_detections(str(img_file), file_detections, str(annotated_path))

    print('\n[4/4] Сохранение результатов...')

    json_path = output_path / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f'  ✓ JSON: {json_path}')

    csv_path = output_path / 'results.csv'
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([
            'ID', 'Файл', 'Тип техники', 'Производитель', 'Модель',
            'Гос. номер', 'Номер читается', 'Уверенность детекции',
            'Уверенность номера', 'ROI номера'
        ])
        for r in all_results:
            writer.writerow([
                r['id'],
                r['filename'],
                r['vehicle_type'],
                r['manufacturer'],
                r['model'],
                r['license_plate'],
                'Да' if r['plate_readable'] else 'Нет',
                f"{r['detection_confidence']:.0%}",
                f"{r['plate_confidence']:.0%}",
                r.get('plate_source') or '',
            ])
    print(f'  ✓ CSV: {csv_path}')

    readable = sum(1 for r in all_results if r['plate_readable'])
    print(f"\n{'=' * 60}")
    print('ИТОГО:')
    print(f'  Обработано изображений: {len(image_files)}')
    print(f'  Обнаружено ТС: {len(all_results)}')
    print(f'  Номеров распознано: {readable}/{len(all_results)}')
    print(f'  Результаты: {output_path}')
    print(f"{'=' * 60}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Распознавание техники и гос. номеров')
    parser.add_argument('--input', '-i', default='data/Auto', help='Папка с изображениями')
    parser.add_argument('--output', '-o', default='results', help='Папка для результатов')
    args = parser.parse_args()

    process_images(args.input, args.output)
