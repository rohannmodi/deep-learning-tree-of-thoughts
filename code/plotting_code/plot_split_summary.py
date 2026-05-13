import matplotlib.pyplot as plt
import numpy as np

def create_split_plots():
    plt.style.use('dark_background')
    tasks = ['Game of 24', 'GSM8K', 'MATH']
    
    # ---------------------------------------------------------
    # PLOT 1: Accuracy (Compact)
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    fig1.patch.set_facecolor('#080808')
    ax1.set_facecolor('#080808')
    
    acc_tot = [46.7, 100.0, 88.0]
    bars1 = ax1.bar(tasks, acc_tot, color='#0a4d4a', edgecolor='#1abc9c', width=0.6, zorder=3)
    
    ax1.set_title('Final ToT Accuracy', fontsize=14, fontweight='bold', color='#1abc9c', pad=15)
    ax1.set_ylabel('Accuracy (%)', color='#cdd6f4')
    ax1.set_ylim(0, 115)
    ax1.yaxis.grid(True, color='#222', linestyle='--')
    
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.1f}%', 
                ha='center', color='#1abc9c', fontweight='bold')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.savefig('accuracy_summary_small.png', dpi=300, bbox_inches='tight', transparent=True)

    # ---------------------------------------------------------
    # PLOT 2: Resource Cost (Compact Grouped)
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.patch.set_facecolor('#080808')
    ax2.set_facecolor('#080808')
    
    times = [51.5, 11.9, 297.0]
    calls = [68.5, 6.0, 154.0]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    b_times = ax2.bar(x - width/2, times, width, label='Time (s)', color='#5a3a0a', edgecolor='#e67e22', zorder=3)
    b_calls = ax2.bar(x + width/2, calls, width, label='API Calls', color='#3a0a5a', edgecolor='#9b59b6', zorder=3)
    
    ax2.set_title('Resource Cost Overhead', fontsize=14, fontweight='bold', color='white', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks)
    ax2.set_ylabel('Value', color='#cdd6f4')
    ax2.yaxis.grid(True, color='#222', linestyle='--')
    ax2.legend(frameon=False, labelcolor='white')
    
    # Value labels
    for bar in b_times:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 5, f'{h:.0f}s', 
                ha='center', color='#e67e22', fontweight='bold', fontsize=9)
    for bar in b_calls:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 5, f'{h:.0f}', 
                ha='center', color='#9b59b6', fontweight='bold', fontsize=9)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.savefig('cost_summary_small.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Small split plots saved.")

if __name__ == '__main__':
    create_split_plots()
