import matplotlib.pyplot as plt
import numpy as np

def create_math_level_cost_plot():
    plt.style.use('default')
    levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
    
    # --- DATA (From our Benchmarks) ---
    # Avg API Calls per level
    calls = [1.0, 1.0, 10.5, 45.2, 35.8]
    # Avg Latency (s) per level
    times = [2.2, 2.4, 25.1, 78.5, 72.7]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    
    x = np.arange(len(levels))
    width = 0.35
    
    # 1. API Calls (Red)
    ax1.set_ylabel('Avg API Calls', color='#c0392b', fontsize=12, fontweight='bold')
    bars1 = ax1.bar(x - width/2, calls, width, label='API Calls', color='#c0392b', edgecolor='black', zorder=3)
    ax1.tick_params(axis='y', labelcolor='#c0392b')
    
    # 2. Latency (Orange) - Secondary Axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Latency (s)', color='#d35400', fontsize=12, fontweight='bold')
    bars2 = ax2.bar(x + width/2, times, width, label='Latency (s)', color='#d35400', edgecolor='black', zorder=3)
    ax2.tick_params(axis='y', labelcolor='#d35400')
    
    # Styling
    ax1.set_title('MATH Dataset: Cost Scaling by Difficulty', fontsize=16, fontweight='bold', color='black', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels, fontweight='bold', color='black')
    ax1.yaxis.grid(True, color='#eee', linestyle='--')
    
    # Hide top/right spines
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}', ha='center', color='black', fontweight='bold', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.0f}s', ha='center', color='black', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig('math_cost_scaling.png', dpi=300, bbox_inches='tight')
    print("MATH cost scaling plot saved.")

if __name__ == '__main__':
    create_math_level_cost_plot()
