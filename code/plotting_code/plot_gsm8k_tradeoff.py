import matplotlib.pyplot as plt
import numpy as np

def create_tradeoff_plot():
    # Setup styling
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1e1e2e')
    
    # Common Data
    methods = ['Single-Shot', 'Chain of Thought', 'Tree of Thoughts']
    
    # ---------------------------------------------------------
    # LEFT PLOT: Accuracy Scaling (The Benefit)
    # ---------------------------------------------------------
    ax1.set_facecolor('#1e1e2e')
    accuracy = [85.0, 90.0, 100.0]
    acc_colors = ['#4facfe', '#00f2fe', '#00c6ff']
    
    bars1 = ax1.bar(methods, accuracy, color=acc_colors, width=0.5, zorder=3)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#cdd6f4')
    ax1.spines['bottom'].set_color('#cdd6f4')
    ax1.tick_params(colors='#cdd6f4', labelsize=11)
    ax1.set_ylim(0, 110)
    ax1.yaxis.grid(True, color='#313244', linestyle='dashed', alpha=0.7, zorder=0)
    ax1.set_axisbelow(True)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='white')
                    
    ax1.set_title('The Benefit: Accuracy Scaling', fontsize=16, fontweight='bold', color='white', pad=15)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, color='#cdd6f4')

    # ---------------------------------------------------------
    # RIGHT PLOT: Compute Cost (The Trade-off)
    # ---------------------------------------------------------
    ax2.set_facecolor('#1e1e2e')
    times = [1.6, 2.1, 11.9]
    calls = [1, 1, 6]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bar_times = ax2.bar(x - width/2, times, width, label='Time (Seconds)', color='#ff7eb3', zorder=3)
    bar_calls = ax2.bar(x + width/2, calls, width, label='API Calls', color='#fee140', zorder=3)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#cdd6f4')
    ax2.spines['bottom'].set_color('#cdd6f4')
    ax2.tick_params(colors='#cdd6f4', labelsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylim(0, 14)
    ax2.yaxis.grid(True, color='#313244', linestyle='dashed', alpha=0.7, zorder=0)
    ax2.set_axisbelow(True)
    
    for bar in bar_times:
        height = bar.get_height()
        ax2.annotate(f'{height}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')
                    
    for bar in bar_calls:
        height = bar.get_height()
        ax2.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')
                    
    ax2.set_title('The Cost: Compute Overhead', fontsize=16, fontweight='bold', color='white', pad=15)
    ax2.legend(loc='upper left', frameon=False, labelcolor='white')

    # Add a main title for the whole figure
    plt.suptitle('GSM8K Evaluation: The Tree of Thoughts Trade-off', 
                 fontsize=20, fontweight='bold', color='white', y=1.05)
                 
    plt.tight_layout()
    
    output_filename = 'gsm8k_tradeoff_plot.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Plot saved to: {output_filename}")

if __name__ == '__main__':
    create_tradeoff_plot()
