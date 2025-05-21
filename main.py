"""
免费文本分析工具 - 单文件版
需安装依赖：pip install jieba snownlp textblob nltk matplotlib reportlab
"""
import re
import os
import csv
import platform
import matplotlib.pyplot as plt
from collections import Counter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import jieba
from snownlp import SnowNLP
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

'''
# 初始化NLTK（首次运行需要下载数据）
try:
    from nltk.data import find

    find('sentiment/vader_lexicon')
except LookupError:
    print("首次使用需要下载NLTK数据，请执行：")
    print("python -m nltk.downloader vader_lexicon")
    exit()
'''

# =======================
# 核心分析模块
# =======================
def detect_language(text):
    """自动检测文本语言"""
    return 'zh' if re.search(r'[\u4e00-\u9fa5]', text) else 'en'


def tokenize_text(text, language):
    """文本分词处理"""
    if language == 'zh':
        return list(jieba.cut(text))
    else:
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def analyze_sentiment(text, language):
    """多语言情感分析"""
    analysis = {
        'score': 0.5,
        'polarity': 'neutral',
        'intensity': {'positive': 0, 'negative': 0, 'neutral': 0},
        'keywords': []
    }

    try:
        if language == 'zh':
            s = SnowNLP(text)
            analysis['score'] = s.sentiments
            # 提取中文情感关键词
            analysis['keywords'] = [(word, tag) for word, tag in s.tags
                                    if tag in ['a', 'ad', 'ag', 'an']]
        else:
            # 英文双重分析
            blob = TextBlob(text)
            sia = SentimentIntensityAnalyzer()
            vader = sia.polarity_scores(text)

            # 综合评分算法
            analysis['score'] = (blob.sentiment.polarity + vader['compound']) / 2
            analysis['intensity'] = {
                'positive': vader['pos'],
                'negative': vader['neg'],
                'neutral': vader['neu']
            }
            # 提取英文关键词
            analysis['keywords'] = [(word, tag) for word, tag in blob.tags
                                    if tag.startswith('JJ')]

        # 判断情感极性
        if analysis['score'] > 0.6:
            analysis['polarity'] = 'positive'
        elif analysis['score'] < 0.4:
            analysis['polarity'] = 'negative'

    except Exception as e:
        print(f"情感分析错误: {str(e)}")

    return analysis


def analyze_file(file_path):
    """主分析函数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 基础信息统计
        stats = {
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'char_count': len(content),
            'line_count': len(content.splitlines()),
            'language': detect_language(content),
            'words': [],
            'sentiment': None
        }

        # 分词处理
        stats['words'] = tokenize_text(content, stats['language'])
        stats['word_count'] = len(stats['words'])
        stats['word_freq'] = Counter(stats['words'])

        # 情感分析
        stats['sentiment'] = analyze_sentiment(content, stats['language'])

        return stats

    except Exception as e:
        print(f"分析失败: {str(e)}")
        return None


# =======================
# 可视化模块
# =======================
def generate_charts(stats):
    """生成分析图表"""
    if not stats or stats['word_count'] == 0:
        return

    # 词频图表
    top_words = stats['word_freq'].most_common(10)
    labels = [w[0] for w in top_words]
    values = [w[1] for w in top_words]

    plt.figure(figsize=(12, 6))

    # 词频柱状图
    plt.subplot(1, 2, 1)
    plt.barh(labels, values)
    plt.title('Top 10 High Frequency words' if stats['language'] == 'zh' else 'Top 10 Words')
    plt.xlabel('Occurrences' if stats['language'] == 'zh' else 'Count')

    # 情感分析图
    plt.subplot(1, 2, 2)
    if stats['language'] == 'zh':
        plt.pie([stats['sentiment']['score'], 1 - stats['sentiment']['score']],
                labels=['positive', 'negative'], autopct='%1.1f%%')
    else:
        intensities = stats['sentiment']['intensity']
        plt.pie([intensities['positive'], intensities['negative'], intensities['neutral']],
                labels=['Positive', 'Negative', 'Neutral'], autopct='%1.1f%%')
    plt.title('Emotional distribution')

    plt.tight_layout()
    plt.savefig('analysis_result.png')
    print("\n图表已保存为: analysis_result.png")


# =======================
# 报告导出模块
# =======================
def export_report(stats, format='csv'):
    """导出分析报告"""
    if not stats:
        return

    # CSV导出
    if format == 'csv':
        with open('report.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['项目', '值'])
            writer.writerows([
                ['文件名', stats['file_name']],
                ['文件大小', f"{stats['file_size']} 字节"],
                ['字符数', stats['char_count']],
                ['行数', stats['line_count']],
                ['词汇量', stats['word_count']],
                ['情感得分', f"{stats['sentiment']['score']:.2f}"],
                ['情感倾向', stats['sentiment']['polarity']]
            ])
        print("CSV报告已导出: report.csv")

    # PDF导出
    elif format == 'pdf':
        pdf = canvas.Canvas("report.pdf", pagesize=letter)
        pdf.setFont("Helvetica", 12)

        y = 750
        pdf.drawString(50, y, "文本分析报告")
        y -= 30

        for item in [
            ('文件名', stats['file_name']),
            ('文件大小', f"{stats['file_size']} 字节"),
            ('字符数', stats['char_count']),
            ('行数', stats['line_count']),
            ('词汇量', stats['word_count']),
            ('情感得分', f"{stats['sentiment']['score']:.2f}"),
            ('情感倾向', stats['sentiment']['polarity'])
        ]:
            pdf.drawString(50, y, f"{item[0]}: {item[1]}")
            y -= 20

        pdf.drawImage('analysis_result.png', 50, y - 200, width=400, height=150)
        pdf.save()
        print("PDF报告已导出: report.pdf")


# =======================
# 主程序
# =======================
if __name__ == "__main__":
    # 检查操作系统编码
    if platform.system() == 'Windows':
        import sys

        sys.stdout.reconfigure(encoding='utf-8')

    # 文件输入
    file_path = input("请输入文本文件路径: ").strip()

    # 执行分析
    print("\n正在分析...")
    result = analyze_file(file_path)

    if result:
        # 显示基础报告
        print(f"\n{' 分析结果 ':=^40}")
        print(f"语言类型: {'中文' if result['language'] == 'zh' else '英文'}")
        print(f"情感倾向: {result['sentiment']['polarity']} ({result['sentiment']['score']:.2f})")

        # 生成图表
        generate_charts(result)

        # 导出报告
        choice = input("\n导出报告格式 (csv/pdf/跳过): ").lower()
        if choice in ['csv', 'pdf']:
            export_report(result, choice)

        print("\n分析完成！")