
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

class MAAcquisitionStrategy:
    def __init__(self):
        self.target_categories = {
            'AI_Software_Platforms': {
                'focus': 'Companies with proven ML platforms for manufacturing',
                'examples': ['Data analytics startups', 'MLOps platforms', 'Predictive maintenance SaaS'],
                'strategic_rationale': 'Accelerate software development by 2-3 years',
                'valuation_range': '$50-200M',
                'integration_timeline': '6-12 months'
            },
            'Specialized_AI_Algorithms': {
                'focus': 'Boutique AI firms with semiconductor-specific algorithms',
                'examples': ['Yield optimization startups', 'Test pattern generation AI', 'Anomaly detection specialists'],
                'strategic_rationale': 'Acquire proprietary algorithms and domain expertise',
                'valuation_range': '$20-80M',
                'integration_timeline': '3-9 months'
            },
            'Complementary_Hardware': {
                'focus': 'Companies with AI-accelerated test hardware',
                'examples': ['FPGA test companies', 'GPU-accelerated measurement firms'],
                'strategic_rationale': 'Enhance real-time inference capabilities',
                'valuation_range': '$100-300M',
                'integration_timeline': '12-18 months'
            },
            'Data_Infrastructure': {
                'focus': 'Companies specializing in industrial data management',
                'examples': ['Time-series database companies', 'Edge computing platforms'],
                'strategic_rationale': 'Build foundation for fleet learning network',
                'valuation_range': '$30-120M',
                'integration_timeline': '6-12 months'
            }
        }

        self.evaluation_criteria = {
            'technology_assessment': {
                'ip_portfolio': 0.15,
                'algorithm_sophistication': 0.20,
                'product_maturity': 0.15,
                'scalability': 0.10
            },
            'business_assessment': {
                'customer_base': 0.10,
                'revenue_traction': 0.08,
                'growth_trajectory': 0.07,
                'profitability': 0.05
            },
            'strategic_assessment': {
                'team_expertise': 0.05,
                'culture_fit': 0.03,
                'integration_complexity': 0.02
            }
        }

    def evaluate_acquisition_target(self, target_company, strategic_fit):
        """Return scored breakdown and recommendation"""
        total_score = 0
        per_criterion = []
        max_score = 0
        for category, criteria in self.evaluation_criteria.items():
            for criterion, weight in criteria.items():
                raw = strategic_fit.get(criterion, 0)  # expected 0..1
                score = raw * weight * 100
                per_criterion.append({
                    'criterion': criterion,
                    'category': category,
                    'weight': weight,
                    'raw_score': raw,
                    'weighted_score': score
                })
                total_score += score
                max_score += 100 * weight

        df = pd.DataFrame(per_criterion).sort_values('weighted_score', ascending=False)
        return {
            'company': target_company,
            'total_score': total_score,
            'max_score': max_score,
            'per_criterion_df': df,
            'acquisition_recommendation': self._get_recommendation(total_score),
            'priority_level': self._get_priority(total_score)
        }

    def _get_recommendation(self, score):
        if score >= 80:
            return "STRONG ACQUIRE - High strategic fit"
        elif score >= 65:
            return "CONSIDER ACQUISITION - Good fit with some risks"
        elif score >= 50:
            return "EVALUATE FURTHER - Marginal strategic value"
        else:
            return "PASS - Low strategic alignment"

    def _get_priority(self, score):
        if score >= 80:
            return "IMMEDIATE"
        elif score >= 65:
            return "HIGH"
        elif score >= 50:
            return "MEDIUM"
        else:
            return "LOW"

    # -------------------------
    # Visualization helpers
    # -------------------------
    def plot_target_category_pie(self):
        """Show a pie of target categories and integration timelines (counts)"""
        keys = list(self.target_categories.keys())
        # For the pie we just demonstrate equal weighting of categories by count
        sizes = np.ones(len(keys))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.pie(sizes, labels=keys, autopct='%1.0f%%', startangle=140)
        ax.set_title('M&A Target Category Distribution (example view)')
        plt.tight_layout()
        return fig

    def plot_criterion_bar(self, per_criterion_df):
        """Bar plot of weighted score per criterion"""
        df = per_criterion_df.copy()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='weighted_score', y='criterion', data=df, hue='category', dodge=False, ax=ax)
        ax.set_xlabel('Weighted Score (0-100)')
        ax.set_ylabel('')
        ax.set_title(f'Per-criterion Weighted Scores for Target')
        ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig

    def plot_score_radar(self, per_criterion_df):
        """Radar-like polygon that groups criteria by category (averaged)"""
        df = per_criterion_df.copy()
        # aggregate by category
        agg = df.groupby('category')['raw_score'].mean()
        labels = agg.index.tolist()
        values = agg.values.tolist()
        # complete the loop
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(angles * 180/np.pi, labels)
        ax.set_title('Average Raw Score by Category (radar view)')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        return fig

# -------------------------
# Example usage (your TestAI Labs)
# -------------------------
if __name__ == "__main__":
    strategy = MAAcquisitionStrategy()
    testai_labs_profile = {
        'ip_portfolio': 0.9,
        'algorithm_sophistication': 0.8,
        'product_maturity': 0.6,
        'scalability': 0.7,
        'customer_base': 0.4,
        'revenue_traction': 0.3,
        'growth_trajectory': 0.8,
        'profitability': 0.2,
        'team_expertise': 0.9,
        'culture_fit': 0.7,
        'integration_complexity': 0.6
    }
    evaluation = strategy.evaluate_acquisition_target("TestAI Labs", testai_labs_profile)
    print(f"Overall Score: {evaluation['total_score']:.1f}/100")
    print(f"Recommendation: {evaluation['acquisition_recommendation']}")
    print(f"Priority: {evaluation['priority_level']}")

    # Plots
    fig1 = strategy.plot_target_category_pie()
    fig2 = strategy.plot_criterion_bar(evaluation['per_criterion_df'])
    fig3 = strategy.plot_score_radar(evaluation['per_criterion_df'])
    plt.show()