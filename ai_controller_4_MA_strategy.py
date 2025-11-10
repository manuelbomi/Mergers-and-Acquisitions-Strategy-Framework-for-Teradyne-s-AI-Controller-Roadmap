# ===============================================================
# Partnership vs Acquisition Decision Framework + Integration Plan Visualization
# ===============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# ===============================================================
# 1. Partnership vs Acquisition Decision Framework
# ===============================================================
class PartnershipVsAcquisition:
    """Evaluate strategic choice between partnership and acquisition."""

    def __init__(self):
        """Define the decision framework and criteria."""
        self.decision_framework = {
            'strategic_importance': {
                'question': 'How critical is this capability to our long-term strategy?',
                'acquisition_threshold': 'Core to competitive advantage',
                'partnership_threshold': 'Complementary but not essential'
            },
            'time_to_market': {
                'question': 'How quickly do we need this capability?',
                'acquisition_threshold': 'Immediate need (<12 months)',
                'partnership_threshold': 'Can wait 18-24 months'
            },
            'ip_control': {
                'question': 'Do we need exclusive control of the IP?',
                'acquisition_threshold': 'Must own the IP',
                'partnership_threshold': 'Can license or share IP'
            },
            'integration_complexity': {
                'question': 'How difficult is integration?',
                'acquisition_threshold': 'Moderate complexity acceptable',
                'partnership_threshold': 'High complexity or cultural mismatch'
            },
            'market_position': {
                'question': 'What is the target company market position?',
                'acquisition_threshold': 'Emerging leader in niche',
                'partnership_threshold': 'Established player too expensive to acquire'
            }
        }

    # ---------------------------------------------------------------
    # Evaluation Logic
    # ---------------------------------------------------------------
    def evaluate_strategy(self, capability_assessment):
        """Score and compare partnership vs acquisition options."""
        acquisition_score = 0
        partnership_score = 0

        # Scoring rules
        scoring = {
            'strategic_importance': {'acquisition': 2, 'partnership': 0},
            'time_to_market': {'acquisition': 2, 'partnership': 0},
            'ip_control': {'acquisition': 2, 'partnership': 0},
            'integration_complexity': {'acquisition': -1, 'partnership': 1},
            'market_position': {'acquisition': 1, 'partnership': 1}
        }

        per_factor = []
        for factor, assessment in capability_assessment.items():
            if assessment == 'acquisition_aligned':
                acq_points = scoring[factor]['acquisition']
                part_points = scoring[factor]['partnership']
            elif assessment == 'partnership_aligned':
                acq_points = scoring[factor]['partnership']
                part_points = scoring[factor]['acquisition']
            else:
                acq_points = 0
                part_points = 0

            acquisition_score += acq_points
            partnership_score += part_points

            per_factor.append({
                'factor': factor,
                'assessment': assessment,
                'acq_points': acq_points,
                'part_points': part_points
            })

        per_df = pd.DataFrame(per_factor)
        decision = "ACQUISITION" if acquisition_score > partnership_score else "STRATEGIC PARTNERSHIP"
        return decision, acquisition_score, partnership_score, per_df

    # ---------------------------------------------------------------
    # Visualization Methods
    # ---------------------------------------------------------------
    def plot_decision_bar(self, acq_score, part_score):
        """Simple bar comparison of acquisition vs partnership total scores."""
        fig, ax = plt.subplots(figsize=(6, 3))
        df = pd.DataFrame({'Score': [acq_score, part_score]}, index=['Acquisition', 'Partnership'])
        df.plot(kind='bar', legend=False, ax=ax, color=['tab:blue', 'tab:orange'])
        ax.set_ylabel('Score')
        ax.set_title('Acquisition vs Partnership Decision Score')
        plt.xticks(rotation=0)
        plt.tight_layout()
        return fig

    def plot_decision_matrix(self, per_df):
        """Heatmap showing factor-by-factor points for each option."""
        pivot = per_df.set_index('factor')[['acq_points', 'part_points']]
        fig, ax = plt.subplots(figsize=(6, max(2, 0.6 * len(pivot))))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap='coolwarm', ax=ax, cbar=False)
        ax.set_title('Decision Points by Strategic Factor')
        plt.tight_layout()
        return fig


# ===============================================================
# 2. Integration Plan Visualization (100-Day Gantt-like Chart)
# ===============================================================
def plot_integration_gantt(plan):
    """
    Display a simple horizontal timeline (Day 0–100) for integration activities.
    Each phase is represented by a colored bar.
    """
    # Define day ranges for each phase
    mapping = {'day_0_30': (0, 30), 'day_31_60': (31, 60), 'day_61_100': (61, 100)}

    fig, ax = plt.subplots(figsize=(10, 3))
    y = 0

    for key, tasks in plan.items():
        start, end = mapping[key]
        ax.broken_barh([(start, end - start + 1)], (y, 0.8), facecolors='tab:blue', alpha=0.6)
        ax.text(start + 2, y + 0.3, key.replace('day_', '').replace('_', ' to '),
                va='center', color='white', fontsize=10, fontweight='bold')
        y += 1

    ax.set_xlabel('Days')
    ax.set_yticks([])
    ax.set_xlim(0, 105)
    ax.set_title('100-Day Integration Plan Overview (Gantt-style)')
    plt.tight_layout()
    return fig


# ===============================================================
# Example Execution
# ===============================================================
if __name__ == "__main__":
    # --- Evaluate a Capability (Edge AI Example) ---
    edge_ai_assessment = {
        'strategic_importance': 'acquisition_aligned',
        'time_to_market': 'acquisition_aligned',
        'ip_control': 'acquisition_aligned',
        'integration_complexity': 'partnership_aligned',
        'market_position': 'acquisition_aligned'
    }

    advisor = PartnershipVsAcquisition()
    recommendation, acq_score, part_score, per_df = advisor.evaluate_strategy(edge_ai_assessment)

    print("\n--- Strategic Evaluation: Edge AI Capability ---")
    print(f"Recommendation: {recommendation}")
    print(f"Acquisition Score: {acq_score}")
    print(f"Partnership Score: {part_score}\n")
    print(per_df)

    # Visualize Decision Outcome
    fig1 = advisor.plot_decision_bar(acq_score, part_score)
    fig2 = advisor.plot_decision_matrix(per_df)

    # --- Integration Plan Visualization ---
    integration_plan = {
        'day_0_30': [
            'Form joint integration team',
            'Align on combined product roadmap',
            'Begin technology integration',
            'Communicate with customers'
        ],
        'day_31_60': [
            'Integrate key algorithms into AI Controller',
            'Cross-train engineering teams',
            'Develop joint go-to-market materials',
            'Identify quick-win customer deployments'
        ],
        'day_61_100': [
            'Launch first integrated product release',
            'Achieve first joint customer success',
            'Measure integration KPIs',
            'Refine long-term integration plan'
        ]
    }

    fig3 = plot_integration_gantt(integration_plan)

    # Show all plots
    plt.show()
