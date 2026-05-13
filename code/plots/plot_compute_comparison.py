import matplotlib.pyplot as plt
import numpy as np

def create_compute_plot():
    # Setup styling - High contrast, premium colors
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor('#0f111a') # Modern deep space background
    
    tasks = ['Game of 24', 'GSM8K', 'MATH']
    
    # --- Data for Compute Cost ---
    # Avg Time in Seconds
    times = [51.5, 11.9, 297.0]
    # Avg API Calls 
    calls = [68.5, 6.0, 154.0]
    
    # 1. Left Panel: Latency (Execution Time)
    ax1.set_facecolor('#0f111a')
    # Use a vibrant Rose/Pink gradient-like feel
    bars1 = ax1.bar(tasks, times, color='#ff007c', alpha=0.85, edgecolor='white', linewidth=1.2, zorder=3)
    
    ax1.set_title('Inference Latency (Seconds)', fontsize=18, fontweight='bold', color='white', pad=20)
    ax1.set_ylabel('Time (s)', fontsize=14, color='#cdd6f4')
    ax1.yaxis.grid(True, color='#2e3244', linestyle='--', alpha=0.6, zorder=0)
    ax1.tick_params(labelsize=12)
    
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 5, f'{h:.1f}s', 
                ha='center', va='bottom', color='#ff007c', fontweight='bold', fontsize=13)

    # 2. Right Panel: API Volume (Total Queries)
    ax2.set_facecolor('#0f111a')
    # Use a Neon Yellow / Amber for API Calls
    bars2 = ax2.bar(tasks, calls, color='#ffee00', alpha=0.85, edgecolor='white', linewidth=1.2, zorder=3)
    
    ax2.set_title('API Volume (Total Calls)', fontsize=18, fontweight='bold', color='white', pad=20)
    ax2.set_ylabel('Total Calls', fontsize=14, color='#cdd6f4')
    ax2.yaxis.grid(True, color='#2e3244', linestyle='--', alpha=0.6, zorder=0)
    ax2.tick_params(labelsize=12)
    
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 3, f'{h:.0f}', 
                ha='center', va='bottom', color='#ffee00', fontweight='bold', fontsize=13)

    # Styling cleanup
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444b6a')
        ax.spines['bottom'].set_color('#444b6a')

    plt.suptitle('The Compute Burden of Deliberation', 
                 fontsize=24, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    plt.savefig('compute_costs_final.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Plot saved as compute_costs_final.png")

if __name__ == '__main__':
    create_compute_plot()
