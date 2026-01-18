from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
import csv
from pathlib import Path

app = Flask(__name__)
app.config['FL_DATA_DIR'] = Path(__file__).parent
CSV_FILE = app.config['FL_DATA_DIR'] / 'manual_check.csv'

def get_all_pairs():
    """获取所有数据库对"""
    pairs = []
    data_dir = app.config['FL_DATA_DIR']
    for item in data_dir.iterdir():
        if item.is_dir() and '_' in item.name:
            parts = item.name.split('_')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                config_file = item / 'config.json'
                if config_file.exists():
                    pairs.append(item.name)
    return sorted(pairs)

def load_config(pair_id):
    """加载指定数据库对的config.json"""
    config_file = app.config['FL_DATA_DIR'] / pair_id / 'config.json'
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def load_annotations():
    """加载已有的标注结果和指标"""
    annotations = {}
    metrics = {}
    if CSV_FILE.exists():
        with open(CSV_FILE, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair_id = row.get('pair_id', '').strip()
                if not pair_id:
                    continue
                
                label = row.get('is_valid_pair', row.get('should_federate', row.get('is_link', ''))).strip().lower()  # 兼容旧字段名
                # 兼容旧的yes/no格式，转换为true/false
                if label == 'yes':
                    label = 'true'
                elif label == 'no':
                    label = 'false'
                annotations[pair_id] = label
                
                # 加载指标（支持空值，表示未标注）
                column_alignment = row.get('column_alignment_good', '').strip().lower()
                label_reasonable_val = row.get('label_reasonable', '').strip().lower()
                task_reasonable_val = row.get('task_reasonable', '').strip().lower()
                is_related_val = row.get('is_related', row.get('is_correlated', '')).strip().lower()  # 兼容旧字段名
                
                metrics[pair_id] = {
                    'column_alignment_good': True if column_alignment == 'true' else (False if column_alignment == 'false' else None),
                    'label_reasonable': True if label_reasonable_val == 'true' else (False if label_reasonable_val == 'false' else None),
                    'task_reasonable': True if task_reasonable_val == 'true' else (False if task_reasonable_val == 'false' else None),
                    'is_related': True if is_related_val == 'true' else (False if is_related_val == 'false' else None)
                }
    return annotations, metrics

def save_annotation(pair_id, is_valid_pair):
    """保存标注结果到CSV"""
    annotations, metrics = load_annotations()
    annotations[pair_id] = 'true' if is_valid_pair else 'false'
    
    # 写入CSV文件（保留所有列）
    all_pairs = set(annotations.keys()) | set(metrics.keys())
    
    with open(CSV_FILE, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['pair_id', 'is_valid_pair', 'column_alignment_good', 'label_reasonable', 
                     'task_reasonable', 'is_related']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for pid in sorted(all_pairs):
            metric_data = metrics.get(pid, {})
            row = {
                'pair_id': pid,
                'is_valid_pair': annotations.get(pid, ''),
                    'column_alignment_good': 'true' if metric_data.get('column_alignment_good') is True else ('false' if metric_data.get('column_alignment_good') is False else ''),
                    'label_reasonable': 'true' if metric_data.get('label_reasonable') is True else ('false' if metric_data.get('label_reasonable') is False else ''),
                    'task_reasonable': 'true' if metric_data.get('task_reasonable') is True else ('false' if metric_data.get('task_reasonable') is False else ''),
                    'is_related': 'true' if metric_data.get('is_related') is True else ('false' if metric_data.get('is_related') is False else '')
                }
            writer.writerow(row)

@app.route('/')
def index():
    """主页：显示所有数据库对列表，支持翻页"""
    pairs = get_all_pairs()
    annotations, metrics = load_annotations()
    
    # 统计信息
    total = len(pairs)
    annotated = sum(1 for p in pairs if p in annotations)
    remaining = total - annotated
    
    # 获取分页参数
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)  # 每页50个
    
    # 构建所有对的列表
    all_pairs_list = []
    for pair in pairs:
        is_annotated = pair in annotations
        pair_info = {
            'id': pair,
            'annotated': is_annotated,
            'is_valid_pair': annotations.get(pair, None),
            'metrics': metrics.get(pair, {})
        }
        all_pairs_list.append(pair_info)
    
    # 分页
    total_pages = (total + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_pairs = all_pairs_list[start_idx:end_idx]
    
    return render_template('index.html', 
                         all_pairs=paginated_pairs,
                         total=total,
                         annotated=annotated,
                         remaining=remaining,
                         page=page,
                         per_page=per_page,
                         total_pages=total_pages)

@app.route('/annotate/<pair_id>')
def annotate(pair_id):
    """标注页面：显示指定数据库对的配置"""
    config = load_config(pair_id)
    if config is None:
        return f"Error: Config file not found for {pair_id}", 404
    
    annotations, metrics = load_annotations()
    current_label = annotations.get(pair_id, None)
    pair_metrics = metrics.get(pair_id, {})
    
    # 获取所有pairs，按顺序导航（不再区分已标注和未标注）
    all_pairs = get_all_pairs()
    
    # 在所有对中找到当前位置
    if pair_id in all_pairs:
        current_index = all_pairs.index(pair_id)
        prev_pair = all_pairs[current_index - 1] if current_index > 0 else None
        next_pair = all_pairs[current_index + 1] if current_index < len(all_pairs) - 1 else None
        current_pos = current_index + 1
        total = len(all_pairs)
    else:
        prev_pair = None
        next_pair = None
        current_pos = 0
        total = len(all_pairs)
    
    return render_template('annotate.html',
                         pair_id=pair_id,
                         config=config,
                         current_label=current_label,
                         metrics=pair_metrics,
                         prev_pair=prev_pair,
                         next_pair=next_pair,
                         current_index=current_pos,
                         total=total)

@app.route('/api/save', methods=['POST'])
def save():
    """API：保存标注结果"""
    data = request.json
    pair_id = data.get('pair_id')
    is_valid_pair = data.get('is_valid_pair', data.get('should_federate', data.get('is_link', False)))  # 兼容旧字段名
    
    if not pair_id:
        return jsonify({'error': 'pair_id is required'}), 400
    
    try:
        save_annotation(pair_id, is_valid_pair)
        return jsonify({'success': True, 'message': 'Annotation saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_annotation', methods=['POST'])
def save_annotation_api():
    """保存标注结果（兼容旧API）"""
    return save()

@app.route('/save_metric', methods=['POST'])
def save_metric_api():
    """保存指标标注结果"""
    data = request.json
    pair_id = data.get('pair_id')
    metric_name = data.get('metric_name')
    value = data.get('value')
    
    if not pair_id or not metric_name:
        return jsonify({'success': False, 'error': 'pair_id and metric_name are required'}), 400
    
    try:
        annotations, metrics = load_annotations()
        
        # 更新指标
        if pair_id not in metrics:
            metrics[pair_id] = {}
        metrics[pair_id][metric_name] = value
        
        # 保存到CSV
        all_pairs = set(annotations.keys()) | set(metrics.keys())
        
        with open(CSV_FILE, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['pair_id', 'is_valid_pair', 'column_alignment_good', 'label_reasonable', 
                         'task_reasonable', 'is_correlated']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for pid in sorted(all_pairs):
                metric_data = metrics.get(pid, {})
                row = {
                    'pair_id': pid,
                    'is_valid_pair': annotations.get(pid, ''),
                    'column_alignment_good': 'true' if metric_data.get('column_alignment_good') is True else ('false' if metric_data.get('column_alignment_good') is False else ''),
                    'label_reasonable': 'true' if metric_data.get('label_reasonable') is True else ('false' if metric_data.get('label_reasonable') is False else ''),
                    'task_reasonable': 'true' if metric_data.get('task_reasonable') is True else ('false' if metric_data.get('task_reasonable') is False else ''),
                    'is_related': 'true' if metric_data.get('is_related') is True else ('false' if metric_data.get('is_related') is False else '')
                }
                writer.writerow(row)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
