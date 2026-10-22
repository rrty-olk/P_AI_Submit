"""実測証跡から日本語PDFとMermaid入りMarkdownを生成する。"""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

from inspection.render import load_evidence
from inspection.localization import LABEL_JA, FONT_REGULAR, FONT_BOLD


def main():
    run = Path(Path('output/latest_run.txt').read_text().strip())
    summary, events, poses = load_evidence(run)
    audit = json.loads((run/'audit.json').read_text())
    assert audit['passed']
    meta = summary['model_metadata']
    pdfmetrics.registerFont(TTFont('ReportJP', str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont('ReportJPBold', str(FONT_BOLD)))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyJP', fontName='ReportJP', fontSize=9.2, leading=15,
                              spaceAfter=8, wordWrap='CJK'))
    styles.add(ParagraphStyle(name='HeadJP', fontName='ReportJPBold', fontSize=14, leading=21,
                              spaceAfter=12, wordWrap='CJK', textColor=colors.HexColor('#153552')))
    styles.add(ParagraphStyle(name='SmallJP', fontName='ReportJP', fontSize=7.5, leading=11,
                              spaceAfter=6, wordWrap='CJK'))
    story, markdown = [], []

    def para(value, style='BodyJP'):
        return Paragraph(escape(str(value)), styles[style])

    def add(value, style='BodyJP'):
        story.append(para(value, style))
        prefix = ('# ' if not markdown else '## ') if style == 'HeadJP' else ''
        markdown.append(prefix + str(value))

    def table(rows, widths):
        markdown.append('\n'.join('| ' + ' | '.join(str(x).replace('|', '\\|') for x in row) + ' |'
                         for row in [rows[0], ['---']*len(rows[0]), *rows[1:]]))
        grid = Table([[para(x, 'SmallJP') for x in row] for row in rows], colWidths=widths,
                     repeatRows=1, hAlign='LEFT')
        grid.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e3edf5')),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), .35, colors.HexColor('#b8c7d6')),
            ('LEFTPADDING', (0,0), (-1,-1), 6), ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5)]))
        story.extend([grid, Spacer(1, 5*mm)])

    add('画像検査AIとROS 2による自律仕分けの検証', 'HeadJP')
    add('Physical AI応用1 最終課題 ／ 実測日：2026年10月22日', 'SmallJP')
    add('1．概要と目的', 'HeadJP')
    add('部品画像をAIで検査し、その判定結果に対応する場所へロボットを自律移動させる仮想仕分けを実装した。画像の認識結果をROS 2の速度指令へつなぎ、姿勢フィードバックによって目標への到達を確認することを目的とする。ロボットにはROS 2 Jazzyのturtlesimを使用し、2次元平面上の移動で仕分けを表現する。')
    add('2．画像に基づくAI判定', 'HeadJP')
    add(f"正常・破損・遮蔽の人工ワッシャー画像9枚を、LiquidAI/d1-omni-600Mの公式system_one APIへ入力した。画像と3択の判定文を与え、PASS（正常）、REJECT（破損）、RECHECK（遮蔽）の判定と各選択肢の確率を取得する。推論は{meta['gpu']}上で実行した。正解ラベルは評価専用であり、AIへの入力や移動先の選択には使用しない。")
    names = ['01_intact.png', '02_broken.png', '03_covered.png']
    story.append(Table([[Image(str(Path('data/images')/n), width=43*mm, height=43*mm) for n in names]],
                       colWidths=[56*mm]*3))
    markdown.append(' '.join(f'![{label}](../../data/images/{name})' for label, name in zip(['正常', '破損', '遮蔽'], names)))
    add('図1．AIに入力する画像の例。左から正常・破損・遮蔽。', 'SmallJP')
    add('3．AI判定からROS 2の移動指令への変換', 'HeadJP')
    add('ROS制御ノードは画像ファイル名をHTTP POST /inferで推論サービスへ渡し、JSON形式の判定・確率・推論時間を受け取る。通常はAIの判定を下表の目標座標へ変換する。最大確率が0.60未満の場合は1秒間停止し、再検査の目標へ移動する。')
    table([['AI判定', '仮想仕分け先', '目標座標 (x, y)'],
           ['PASS', '合格', '(8.5, 8.5)'], ['REJECT', '不合格', '(2.0, 8.5)'],
           ['RECHECK', '再検査', '(5.5, 2.0)']], [50*mm, 50*mm, 70*mm])
    add('制御ノードは/turtle1/poseから現在の位置と向きを読み取り、目標との差に応じて並進速度と角速度を計算する。geometry_msgs/Twistを/inspection/cmd_velへ送信し、独立した速度監視ノードを経由して/turtle1/cmd_velへ伝える。位置誤差が0.12以下になったら停止して到達を記録し、次の画像へ進む。')

    story.append(PageBreak())
    add('4．システム構成図', 'HeadJP')
    add('画像の判定結果を目標座標へ変換し、ROS 2の速度指令と姿勢フィードバックによって移動する。目標への到達後に次の画像を検査する。')
    source = Path('docs/system_architecture.mmd').read_text(encoding='utf-8').strip()
    markdown.append('```mermaid\n' + source + '\n```')
    figure = Image('docs/images/system_architecture.png')
    scale = min(170*mm/figure.imageWidth, 172*mm/figure.imageHeight)
    figure.drawWidth, figure.drawHeight = figure.imageWidth*scale, figure.imageHeight*scale
    story.append(figure)
    add('図2．AI判定とROS 2による自律移動の構成。', 'SmallJP')
    add('工夫した点', 'HeadJP')
    add('AI推論とROSの移動制御を分け、判定の完了後に移動を開始する。移動時間を決め打ちせず、実際の位置に基づいて到達を確認してから次の検査へ進むため、複数画像を連続して処理できる。低確信度の判定は再検査へ振り分ける。')
    add('推論待ち中は停止する。姿勢更新が0.5秒以上途絶えた場合、不正な応答を受け取った場合、推論または移動が30秒を超えた場合は停止して処理を中止する。速度監視ノードも指令更新が0.5秒途絶えた場合や非数値の速度指令を検出した場合に停止させる。')

    story.append(PageBreak())
    add('5．AI判定と移動の実測結果', 'HeadJP')
    add(f"9枚を順番に検査し、判定に基づく3種類の目標へ全9件到達した。低確信度による一時停止と再検査への振り分けは{audit['low_confidence_stops']}件だった。表の「AI判定」はモデルの最大確率の選択肢、「移動先」は低確信度処理も反映した振り分け先を示す。")
    rows = [['検査番号', '正解', 'AI判定', '移動先', '確信度', '推論ms']]
    for i, c in enumerate(summary['cases'], 1):
        rows.append([f'{i:02d}', LABEL_JA[c['expected']], LABEL_JA[c['model_choice']], LABEL_JA[c['route']],
                     f"{c['confidence']:.3f}", f"{c['inference_seconds']*1000:.1f}"])
    table(rows, [26*mm, 27*mm, 27*mm, 27*mm, 32*mm, 31*mm])
    table([['測定項目', '実測値'],
           ['選択された目標への到達', f"{audit['case_count']}/9件（目標座標は{audit['distinct_routes']}種類）"],
           ['到達時の最大位置誤差', f"{audit['max_arrival_error']:.6f}（許容値0.12）"],
           ['GPU推論の平均時間', f"{audit['mean_inference_seconds']*1000:.1f}ミリ秒"],
           ['AI判定の正解率／振り分け正解率', f"{summary['raw_accuracy']:.1%}／{audit['route_accuracy']:.1%}"]],
          [85*mm, 85*mm])
    out = Path('output/submission/report.pdf')

    def footer(canvas, doc):
        canvas.setFont('ReportJP', 8)
        canvas.setFillColor(colors.HexColor('#52677d'))
        canvas.drawString(20*mm, 12*mm, 'Physical AI 最終課題 ／ 画像検査と自律仕分け')
        canvas.drawRightString(190*mm, 12*mm, str(doc.page))

    doc = SimpleDocTemplate(str(out), pagesize=(210*mm, 297*mm), leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=18*mm, bottomMargin=20*mm, title='画像検査AIとROS 2による自律仕分けの検証', author='')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    out.with_suffix('.md').write_text('\n\n'.join(markdown)+'\n', encoding='utf-8')
    print(out)
    print(out.with_suffix('.md'))


if __name__ == '__main__':
    main()
