"""
Utility functions for rent prediction project
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fmt
import os


def setup_japanese_font():
    """
    日本語フォントを設定する

    優先順位:
    1. Meiryoフォント (meiryo.ttc)
    2. システムの日本語フォント
    3. DejaVu Sans (フォールバック)
    """
    font_configured = False

    # 1. Meiryoフォントファイルを探す
    font_candidates = [
        'meiryo.ttc',
        './meiryo.ttc',
        '../meiryo.ttc',
        '/usr/share/fonts/truetype/meiryo.ttc',
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                fmt.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = ['Meiryo']
                font_configured = True
                print(f"✅ 日本語フォント設定完了: {font_path}")
                break
            except Exception as e:
                print(f"Warning: {font_path} の読み込みエラー: {e}")

    # 2. システムの日本語フォントを試す
    if not font_configured:
        japanese_fonts = [
            'Meiryo',
            'Yu Gothic',
            'MS Gothic',
            'Hiragino Sans',
            'Noto Sans CJK JP',
            'IPAGothic',
            'TakaoPGothic'
        ]

        available_fonts = [f.name for f in fmt.fontManager.ttflist]

        for font_name in japanese_fonts:
            if font_name in available_fonts:
                plt.rcParams['font.family'] = [font_name]
                font_configured = True
                print(f"✅ 日本語フォント設定完了: {font_name}")
                break

    # 3. フォールバック
    if not font_configured:
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
        print("⚠️ 日本語フォントが見つかりません。デフォルトフォントを使用します。")

    # マイナス記号の文字化け対策
    plt.rcParams['axes.unicode_minus'] = False

    return font_configured


def list_available_japanese_fonts():
    """
    利用可能な日本語フォントのリストを表示
    """
    japanese_keywords = ['Japan', 'Gothic', 'Mincho', 'Meiryo', 'Yu', 'MS', 'IPA', 'Takao', 'Noto', 'CJK', 'Hiragino']

    available_fonts = set()
    for font in fmt.fontManager.ttflist:
        if any(keyword in font.name for keyword in japanese_keywords):
            available_fonts.add(font.name)

    if available_fonts:
        print("利用可能な日本語フォント:")
        for font_name in sorted(available_fonts):
            print(f"  - {font_name}")
    else:
        print("日本語フォントが見つかりませんでした。")

    return sorted(available_fonts)


if __name__ == "__main__":
    print("=" * 60)
    print("日本語フォント設定テスト")
    print("=" * 60)

    setup_japanese_font()

    print("\n" + "=" * 60)
    list_available_japanese_fonts()
    print("=" * 60)
