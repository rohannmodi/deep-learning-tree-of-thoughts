import matplotlib.pyplot as plt
import numpy as np

def create_master_summary():
    # Setup styling
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#1e1e2e')
    
    tasks = ['Game of 24', 'GSM8K', 'MATH']
    
    # --- Data for Accuracy (Panel 1) ---
    # Game of 24: SS=26.7%, ToT=46.7% (Our 15-puzzle sample)
    # GSM8K: SS=85%, ToT=100%
    # MATH: SS=84%, ToT=88%
    ss_acc = [26.7, 85.0, 84.0]
    tot_acc = [46.7, 100.0, 88.0]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    # Panel 1: Accuracy Scaling
    ax1.set_facecolor('#1e1e2e')
    b1 = ax1.bar(x - width/2, ss_acc, width, label='Single-Shot', color='#4facfe', zorder=3)
    b2 = ax1.bar(x + width/2, tot_acc, width, label='Tree of Thoughts', color='#00c6ff', zorder=3)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12, color='#cdd6f4')
    ax1.set_title('Accuracy ROI: Single-Shot vs. ToT', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks)
    ax1.set_ylim(0, 115)
    ax1.legend(loc='upper left', frameon=False)
    ax1.yaxis.grid(True, color='#313244', linestyle='dashed', alpha=0.7, zorder=0)
    
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.1f}%', 
                    ha='center', va='bottom', color='white', fontweight='bold')

    # --- Data for Compute Cost (Panel 2) ---
    # These represent the "Call Multiplier" (ToT Calls / SS Calls)
    # Game of 24: 68.5x
    # GSM8K: 6.0x
    # MATH (Avg per Hard Problem): 154.0x
    cost_multipliers = [68.5, 6.0, 154.0]
    
    ax2.set_facecolor('#1e1e2e')
    b3 = ax2.bar(tasks, cost_multipliers, color=['#ff7eb3', '#fee140', '#ff0000'], width=0.6, zorder=3)
    
    ax2.set_ylabel('Compute Multiplier (API Calls)', fontsize=12, color='#cdd6f4')
    ax2.set_title('The Cost: API Call Multiplier', fontsize=16, fontweight='bold', pad=15)
    ax2.yaxis.grid(True, color='#313244', linestyle='dashed', alpha=0.7, zorder=0)
    ax2.set_ylim(0, 180) # Headroom for labels
    
    for bar in b3:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 5, f'{h:.1f}x', 
                ha='center', va='bottom', color='white', fontweight='bold')
    
    # Overall layout
    plt.suptitle('Tree of Thoughts Benchmark: The Global Trade-off', 
                 fontsize=22, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    
    plt.savefig('master_summary_plot.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Master summary plot saved as master_summary_plot.png")

if __name__ == '__main__':
    create_master_summary()
