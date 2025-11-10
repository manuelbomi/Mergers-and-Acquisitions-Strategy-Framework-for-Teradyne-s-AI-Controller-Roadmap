# AcquisitionFinancialModel with visualizations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

class AcquisitionFinancialModel:
    def __init__(self):
        self.valuation_metrics = {
            'revenue_multiple': [8.0, 12.0],
            'ebitda_multiple': [15.0, 25.0],
            'strategic_premium': 0.3  # 30%
        }

    def calculate_acquisition_valuation(self, target_financials):
        revenue = target_financials['annual_revenue']
        revenue_valuation = [
            revenue * self.valuation_metrics['revenue_multiple'][0],
            revenue * self.valuation_metrics['revenue_multiple'][1]
        ]

        if target_financials['ebitda_margin'] > 0:
            ebitda = revenue * target_financials['ebitda_margin']
            ebitda_valuation = [
                ebitda * self.valuation_metrics['ebitda_multiple'][0],
                ebitda * self.valuation_metrics['ebitda_multiple'][1]
            ]
        else:
            ebitda_valuation = [0, 0]

        strategic_valuation = [
            max(revenue_valuation[0], ebitda_valuation[0]) * (1 + self.valuation_metrics['strategic_premium']),
            max(revenue_valuation[1], ebitda_valuation[1]) * (1 + self.valuation_metrics['strategic_premium'])
        ]

        return {
            'revenue_based_valuation': revenue_valuation,
            'ebitda_based_valuation': ebitda_valuation,
            'strategic_valuation_range': strategic_valuation,
            'recommended_offer': strategic_valuation[0]
        }

    # -------------------------
    # Visualization helpers
    # -------------------------
    def plot_valuation_ranges(self, target_financials, valuation_result):
        """Plot revenue-based, EBITDA-based and strategic valuation ranges as bars"""
        labels = ['Low', 'High']
        rev = valuation_result['revenue_based_valuation']
        ebit = valuation_result['ebitda_based_valuation']
        strat = valuation_result['strategic_valuation_range']

        df = pd.DataFrame({
            'Revenue-based': rev,
            'EBITDA-based': ebit,
            'Strategic': strat
        }, index=labels)

        fig, ax = plt.subplots(figsize=(8, 5))
        df.plot(kind='bar', ax=ax)
        ax.set_ylabel('Valuation ($)')
        ax.set_title(f"Valuation Ranges for {target_financials.get('company_name','Target')}")
        plt.xticks(rotation=0)
        plt.tight_layout()
        return fig

    def plot_multiple_scenarios(self, target_financials, revenue_multiples=None):
        """Show sensitivity: valuation vs. revenue multiple sweep"""
        if revenue_multiples is None:
            revenue_multiples = np.linspace(6, 14, 17)
        revenue = target_financials['annual_revenue']
        premium = self.valuation_metrics['strategic_premium']

        valuations = [(m * revenue) * (1 + premium) for m in revenue_multiples]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(revenue_multiples, valuations, marker='o')
        ax.set_xlabel('Revenue Multiple')
        ax.set_ylabel('Strategic Valuation ($)')
        ax.set_title('Sensitivity: Strategic Valuation vs Revenue Multiple')
        ax.grid(True)
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    target_financials = {
        'company_name': 'DataMind AI',
        'annual_revenue': 8_000_000,
        'revenue_growth_rate': 0.60,
        'ebitda_margin': -0.15,
        'customer_count': 45,
        'key_technology': 'AI-powered test correlation analytics'
    }
    fm = AcquisitionFinancialModel()
    valuation = fm.calculate_acquisition_valuation(target_financials)
    print("Recommended Offer:", f"${valuation['recommended_offer']:,.0f}")

    fig1 = fm.plot_valuation_ranges(target_financials, valuation)
    fig2 = fm.plot_multiple_scenarios(target_financials)
    plt.show()