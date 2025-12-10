import pandas as pd
import numpy as np
import akshare as ak
import requests
import json
from datetime import datetime , timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class SmartMoneyScorer:
    """
    聪明资金综合评分系统
    从融资情况、北向资金、主力资金、技术指标等多维度评估股票
    """

    def __init__(self):
        self.weights = {
            'funding_score': 0.15 ,  # 融资情况
            'north_money_score': 0.20 ,  # 北向资金
            'main_money_score': 0.15 ,  # 主力资金
            'price_volume_score': 0.10 ,  # 价量关系
            'technical_score': 0.20 ,  # 技术指标
            'market_performance_score': 0.10 ,  # 市场表现
            'liquidity_score': 0.10  # 流通盘
        }

    def get_stock_basic_info(self , stock_code):
        """获取股票基本信息"""
        try:
            # 使用akshare获取股票基本信息
            stock_info = ak.stock_individual_info_em(symbol = stock_code)
            return stock_info
        except:
            return None

    def calculate_funding_score(self , stock_code):
        """计算融资情况得分:cite[5]:cite[9]"""
        try:
            # 获取融资融券数据
            margin_data = ak.stock_margin_sse(symbol = stock_code)
            if margin_data.empty:
                return 50

            # 计算融资余额变化率
            latest_margin = margin_data.iloc[ 0 ][ 'rzjyye' ]  # 融资余额
            prev_margin = margin_data.iloc[ 1 ][ 'rzjyye' ] if len(margin_data) > 1 else latest_margin

            if prev_margin > 0:
                margin_change = (latest_margin - prev_margin) / prev_margin
            else:
                margin_change = 0

            # 融资余额增长得分
            if margin_change > 0.1:
                margin_score = 90
            elif margin_change > 0.05:
                margin_score = 75
            elif margin_change > 0:
                margin_score = 60
            elif margin_change > -0.05:
                margin_score = 40
            else:
                margin_score = 20

            return margin_score
        except:
            return 50

    def calculate_north_money_score(self , stock_code):
        """计算北向资金得分:cite[1]"""
        try:
            # 获取北向资金持股数据
            north_data = ak.stock_hsgt_hold_stock_em(market = "北向" , symbol = stock_code)
            if north_data.empty:
                return 50

            # 计算持股比例变化
            latest_hold = north_data.iloc[ 0 ][ '持股数量' ]
            prev_hold = north_data.iloc[ 1 ][ '持股数量' ] if len(north_data) > 1 else latest_hold

            if prev_hold > 0:
                hold_change = (latest_hold - prev_hold) / prev_hold
            else:
                hold_change = 0

            # 北向资金增持得分
            if hold_change > 0.2:
                north_score = 95
            elif hold_change > 0.1:
                north_score = 80
            elif hold_change > 0.05:
                north_score = 70
            elif hold_change > 0:
                north_score = 60
            elif hold_change > -0.05:
                north_score = 45
            elif hold_change > -0.1:
                north_score = 30
            else:
                north_score = 20

            return north_score
        except:
            return 50

    def calculate_main_money_score(self , stock_code):
        """计算主力资金得分:cite[9]"""
        try:
            # 获取主力资金流向数据
            money_flow = ak.stock_individual_fund_flow(stock = stock_code , market = "主板")
            if money_flow.empty:
                return 50

            # 计算主力净流入
            main_net_inflow = money_flow.iloc[ 0 ][ '主力净流入' ]
            turnover = money_flow.iloc[ 0 ][ '换手率' ]

            # 主力资金得分
            if main_net_inflow > 10000000:  # 净流入超1000万
                main_score = 90
            elif main_net_inflow > 5000000:
                main_score = 75
            elif main_net_inflow > 0:
                main_score = 60
            elif main_net_inflow > -5000000:
                main_score = 40
            else:
                main_score = 25

            return main_score
        except:
            return 50

    def calculate_technical_score(self , stock_code):
        """计算技术指标得分:cite[2]:cite[6]"""
        try:
            # 获取历史价格数据
            stock_data = ak.stock_zh_a_hist(symbol = stock_code , period = "daily" ,
                                            start_date = (datetime.now( ) - timedelta(days = 60)).strftime('%Y%m%d') ,
                                            end_date = datetime.now( ).strftime('%Y%m%d'))
            if stock_data.empty:
                return 50

            # 计算布林带
            stock_data[ 'MA20' ] = stock_data[ '收盘' ].rolling(window = 20).mean( )
            stock_data[ 'STD20' ] = stock_data[ '收盘' ].rolling(window = 20).std( )
            stock_data[ 'Upper' ] = stock_data[ 'MA20' ] + 2 * stock_data[ 'STD20' ]
            stock_data[ 'Lower' ] = stock_data[ 'MA20' ] - 2 * stock_data[ 'STD20' ]

            latest_close = stock_data.iloc[ -1 ][ '收盘' ]
            ma20 = stock_data.iloc[ -1 ][ 'MA20' ]
            upper = stock_data.iloc[ -1 ][ 'Upper' ]
            lower = stock_data.iloc[ -1 ][ 'Lower' ]

            # 布林带位置得分:cite[2]
            bb_position = (latest_close - lower) / (upper - lower) if (upper - lower) > 0 else 0.5

            if bb_position > 0.7:
                bb_score = 30  # 接近上轨，可能超买
            elif bb_position > 0.3:
                bb_score = 70  # 中轨附近
            else:
                bb_score = 40  # 接近下轨，可能超卖

            # 均线排列得分
            ma5 = stock_data[ '收盘' ].rolling(window = 5).mean( ).iloc[ -1 ]
            ma10 = stock_data[ '收盘' ].rolling(window = 10).mean( ).iloc[ -1 ]

            if ma5 > ma10 > ma20:
                ma_score = 90
            elif ma5 > ma10:
                ma_score = 70
            elif ma5 > ma20:
                ma_score = 60
            else:
                ma_score = 40

            # SAR指标（简化版）
            recent_high = stock_data[ '最高' ].tail(5).max( )
            recent_low = stock_data[ '最低' ].tail(5).min( )

            if latest_close > recent_high:
                sar_score = 80
            elif latest_close < recent_low:
                sar_score = 30
            else:
                sar_score = 60

            technical_score = (bb_score * 0.3 + ma_score * 0.4 + sar_score * 0.3)
            return technical_score
        except:
            return 50

    def calculate_price_volume_score(self , stock_code):
        """计算价量关系得分:cite[10]"""
        try:
            stock_data = ak.stock_zh_a_hist(symbol = stock_code , period = "daily" ,
                                            start_date = (datetime.now( ) - timedelta(days = 30)).strftime('%Y%m%d') ,
                                            end_date = datetime.now( ).strftime('%Y%m%d'))
            if stock_data.empty:
                return 50

            # 计算量价配合度
            price_change = (stock_data.iloc[ -1 ][ '收盘' ] - stock_data.iloc[ -2 ][ '收盘' ]) / stock_data.iloc[ -2 ][
                '收盘' ]
            volume_change = (stock_data.iloc[ -1 ][ '成交量' ] - stock_data.iloc[ -2 ][ '成交量' ]) / \
                            stock_data.iloc[ -2 ][ '成交量' ]

            # 价涨量增或价跌量缩为健康
            if (price_change > 0 and volume_change > 0) or (price_change < 0 and volume_change < 0):
                pv_score = 80
            elif (price_change > 0 and volume_change < 0) or (price_change < 0 and volume_change > 0):
                pv_score = 40
            else:
                pv_score = 60

            return pv_score
        except:
            return 50

    def calculate_market_performance_score(self , stock_code):
        """计算市场表现得分"""
        try:
            stock_data = ak.stock_zh_a_hist(symbol = stock_code , period = "daily" ,
                                            start_date = (datetime.now( ) - timedelta(days = 60)).strftime('%Y%m%d') ,
                                            end_date = datetime.now( ).strftime('%Y%m%d'))
            if stock_data.empty:
                return 50

            # 计算近期收益率
            current_price = stock_data.iloc[ -1 ][ '收盘' ]
            price_20_days_ago = stock_data.iloc[ -20 ][ '收盘' ] if len(stock_data) >= 20 else stock_data.iloc[ 0 ][
                '收盘' ]

            return_20d = (current_price - price_20_days_ago) / price_20_days_ago

            # 收益率得分
            if return_20d > 0.2:
                performance_score = 90
            elif return_20d > 0.1:
                performance_score = 75
            elif return_20d > 0:
                performance_score = 65
            elif return_20d > -0.1:
                performance_score = 45
            else:
                performance_score = 30

            return performance_score
        except:
            return 50

    def calculate_liquidity_score(self , stock_code):
        """计算流通盘得分:cite[9]"""
        try:
            # 获取流通市值信息
            stock_info = ak.stock_individual_info_em(symbol = stock_code)
            if stock_info.empty:
                return 50

            # 这里简化处理，实际需要获取流通市值
            # 小盘股通常弹性更大，但风险也更高
            market_cap = stock_info.get('总市值' , 0)

            if market_cap < 5000000000:  # 50亿以下
                liquidity_score = 70  # 小盘股，弹性大
            elif market_cap < 20000000000:  # 200亿以下
                liquidity_score = 80  # 中盘股，平衡
            elif market_cap < 100000000000:  # 1000亿以下
                liquidity_score = 65  # 大盘股，稳定
            else:
                liquidity_score = 50  # 超大盘股，弹性小

            return liquidity_score
        except:
            return 50

    def calculate_comprehensive_score(self , stock_code):
        """计算综合评分"""
        scores = {}

        scores[ 'funding_score' ] = self.calculate_funding_score(stock_code)
        scores[ 'north_money_score' ] = self.calculate_north_money_score(stock_code)
        scores[ 'main_money_score' ] = self.calculate_main_money_score(stock_code)
        scores[ 'technical_score' ] = self.calculate_technical_score(stock_code)
        scores[ 'price_volume_score' ] = self.calculate_price_volume_score(stock_code)
        scores[ 'market_performance_score' ] = self.calculate_market_performance_score(stock_code)
        scores[ 'liquidity_score' ] = self.calculate_liquidity_score(stock_code)

        # 计算加权总分
        total_score = 0
        for dimension , score in scores.items( ):
            total_score += score * self.weights[ dimension ]

        scores[ 'total_score' ] = total_score
        scores[ 'stock_code' ] = stock_code

        return scores

    def analyze_multiple_stocks(self , stock_list):
        """分析多只股票"""
        results = [ ]
        for stock in stock_list:
            try:
                score_result = self.calculate_comprehensive_score(stock)
                results.append(score_result)
                print(f"已完成 {stock} 的分析")
            except Exception as e:
                print(f"分析 {stock} 时出错: {str(e)}")
                continue

        # 转换为DataFrame并排序
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('total_score' , ascending = False)

        return df_results

    def generate_radar_chart(self , scores , stock_name):
        """生成雷达图"""
        dimensions = [ '融资情况' , '北向资金' , '主力资金' , '技术指标' ,
                       '价量关系' , '市场表现' , '流通盘' ]

        values = [
            scores[ 'funding_score' ] ,
            scores[ 'north_money_score' ] ,
            scores[ 'main_money_score' ] ,
            scores[ 'technical_score' ] ,
            scores[ 'price_volume_score' ] ,
            scores[ 'market_performance_score' ] ,
            scores[ 'liquidity_score' ]
        ]

        # 闭合雷达图
        values.append(values[ 0 ])
        dimensions.append(dimensions[ 0 ])

        angles = np.linspace(0 , 2 * np.pi , len(dimensions) , endpoint = True)

        fig , ax = plt.subplots(figsize = (10 , 10) , subplot_kw = dict(projection = 'polar'))
        ax.plot(angles , values , 'o-' , linewidth = 2 , label = stock_name)
        ax.fill(angles , values , alpha = 0.25)
        ax.set_thetagrids(angles[ :-1 ] * 180 / np.pi , dimensions[ :-1 ])
        ax.set_ylim(0 , 100)
        ax.grid(True)
        plt.title(f'{stock_name} - 聪明资金综合评分雷达图\n综合得分: {scores[ "total_score" ]:.1f}' ,
                  size = 14 , pad = 20)
        plt.legend(loc = 'upper right' , bbox_to_anchor = (1.3 , 1.1))
        plt.tight_layout( )
        plt.show( )

    def generate_report(self , scores , stock_code):
        """生成详细报告"""
        print("=" * 60)
        print("聪明资金综合评分报告")
        print("=" * 60)
        print(f"股票代码: {stock_code}")
        print(f"分析时间: {datetime.now( ).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"综合得分: {scores[ 'total_score' ]:.2f}")
        print("\n各维度得分详情:")
        print("-" * 40)

        dimension_names = {
            'funding_score': '融资情况' ,
            'north_money_score': '北向资金' ,
            'main_money_score': '主力资金' ,
            'technical_score': '技术指标' ,
            'price_volume_score': '价量关系' ,
            'market_performance_score': '市场表现' ,
            'liquidity_score': '流通盘'
        }

        for dim_en , dim_cn in dimension_names.items( ):
            weight = self.weights[ dim_en ] * 100
            score = scores[ dim_en ]
            print(f"{dim_cn:8}: {score:6.1f}分 (权重:{weight:4.1f}%)")

        print("-" * 40)

        # 投资建议
        total_score = scores[ 'total_score' ]
        if total_score >= 80:
            rating = "★★★★★ 强烈关注"
            suggestion = "资金面和技术面均表现优异，聪明资金持续流入"
        elif total_score >= 70:
            rating = "★★★★☆ 积极关注"
            suggestion = "多个维度表现良好，具备投资价值"
        elif total_score >= 60:
            rating = "★★★☆☆ 一般关注"
            suggestion = "整体表现中等，需关注关键指标变化"
        elif total_score >= 50:
            rating = "★★☆☆☆ 谨慎关注"
            suggestion = "部分维度存在弱势，需要谨慎对待"
        else:
            rating = "★☆☆☆☆ 保持观望"
            suggestion = "多个维度表现不佳，建议观望"

        print(f"投资评级: {rating}")
        print(f"操作建议: {suggestion}")


# 使用示例
def main( ):
    # 创建评分器
    scorer = SmartMoneyScorer( )

    # 要分析的股票列表（示例）
    stock_list = [ '000858' , '600519' , '000333' , '601318' , '600036' ]

    print("开始分析聪明资金综合评分...")

    # 分析多只股票
    results = scorer.analyze_multiple_stocks(stock_list)

    # 显示结果
    print("\n" + "=" * 80)
    print("聪明资金综合评分排行榜")
    print("=" * 80)
    print(results[ [ 'stock_code' , 'total_score' , 'north_money_score' ,
                     'main_money_score' , 'technical_score' ] ].round(2))

    # 对排名第一的股票生成详细报告和雷达图
    if not results.empty:
        best_stock = results.iloc[ 0 ][ 'stock_code' ]
        best_scores = results.iloc[ 0 ].to_dict( )

        print(f"\n表现最佳股票分析: {best_stock}")
        scorer.generate_report(best_scores , best_stock)
        scorer.generate_radar_chart(best_scores , f"股票{best_stock}")

    return results


# 单只股票分析示例
def analyze_single_stock(stock_code):
    """分析单只股票"""
    scorer = SmartMoneyScorer( )

    print(f"开始分析股票 {stock_code}...")
    scores = scorer.calculate_comprehensive_score(stock_code)

    scorer.generate_report(scores , stock_code)
    scorer.generate_radar_chart(scores , f"股票{stock_code}")

    return scores


if __name__ == "__main__":
    # 分析多只股票
    results = main( )

    # 或者分析单只股票
    # scores = analyze_single_stock('000858')