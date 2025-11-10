# TechnologyGapAnalysis with visualizations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

class TechnologyGapAnalysis:
    def __init__(self):
        self.ai_controller_requirements = {
            'core_ml_infrastructure': ['Model training', 'Inference engine', 'Data pipelines'],
            'domain_specific_algorithms': ['Adaptive test', 'Yield prediction', 'Anomaly detection'],
            'edge_compute': ['Real-time inference', 'GPU acceleration', 'Low-latency processing'],
            'data_management': ['Time-series databases', 'Data security', 'Fleet learning'],
            'ui_ux': ['Dashboard visualization', 'Alert systems', 'Report generation']
        }

        self.current_capabilities = {
            'core_ml_infrastructure': 0.4,
            'domain_specific_algorithms': 0.6,
            'edge_compute': 0.3,
            'data_management': 0.5,
            'ui_ux': 0.7
        }

    def identify_acquisition_targets(self):
        gaps = {}
        acquisition_priorities = []
        for category, capability_score in self.current_capabilities.items():
            gap_size = 1.0 - capability_score
            if gap_size > 0.3:
                gaps[category] = {
                    'gap_size': gap_size,
                    'build_vs_buy_analysis': self._build_vs_buy_analysis(category, gap_size),
                    'acquisition_priority': 'HIGH' if gap_size > 0.5 else 'MEDIUM'
                }
                acquisition_priorities.append(category)
        return gaps, acquisition_priorities

    def _build_vs_buy_analysis(self, category, gap_size):
        build_timeline = {
            'core_ml_infrastructure': 24,
            'domain_specific_algorithms': 18,
            'edge_compute': 12,
            'data_management': 15,
            'ui_ux': 9
        }
        acquisition_timeline = 6
        time_advantage = build_timeline.get(category, 12) - acquisition_timeline
        strategic_advantage = "ACQUIRE" if time_advantage > 6 else "BUILD"
        return {
            'build_timeline_months': build_timeline.get(category, 12),
            'acquisition_timeline_months': acquisition_timeline,
            'time_advantage_months': time_advantage,
            'recommendation': strategic_advantage
        }

    # -------------------------
    # Visualization helpers
    # -------------------------
    def plot_capability_bars(self):
        """Bar chart of current capability coverage (0..1) with gaps"""
        df = pd.DataFrame.from_dict(self.current_capabilities, orient='index', columns=['coverage'])
        df = df.sort_values('coverage')
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='coverage', y=df.index, data=df.reset_index().rename(columns={'index':'category'}), ax=ax)
        ax.set_xlabel('Coverage (0..1)')
        ax.set_title('Current Capability Coverage for AI Controller')
        plt.tight_layout()
        return fig

    def plot_gap_heatmap(self):
        """Heatmap of gap sizes across categories (single-dimension shown as heatmap for presentation)"""
        categories = list(self.current_capabilities.keys())
        gap_sizes = [1 - self.current_capabilities[c] for c in categories]
        df = pd.DataFrame([gap_sizes], columns=categories)
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.heatmap(df, annot=True, fmt=".2f", cmap='Reds', cbar=False, ax=ax)
        ax.set_title('Gap Size Heatmap (1 - coverage)')
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    gap_analysis = TechnologyGapAnalysis()
    gaps, priorities = gap_analysis.identify_acquisition_targets()
    print("TOP ACQUISITION PRIORITIES:", priorities)
    fig1 = gap_analysis.plot_capability_bars()
    fig2 = gap_analysis.plot_gap_heatmap()
    plt.show()