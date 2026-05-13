import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

def create_gsm8k_tree_plot():
    # Setup styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 13))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Create directed graph
    G = nx.DiGraph()

    # Define nodes with positions and labels
    # Format: node_id: (x, y, label, color)
    nodes = {
        'Root': (5, 4.5, 'Input:\n3 pens ($2), 4 books ($5).\nPaid with $50. Change?', '#cba6f7'),  # Purple
        
        # Depth 1
        'N1_1': (2, 3.5, 'Step 1:\nPens cost: 3 * $2 = $6\nEval: likely', '#a6e3a1'), # Green
        'N1_2': (5, 3.5, 'Step 1:\nBooks cost: 4 * $5 = $20\nEval: likely', '#f9e2af'), # Yellow (also valid, just exploring)
        'N1_3': (8, 3.5, 'Step 1:\nTotal items: 3 + 4 = 7\nEval: unlikely', '#f38ba8'), # Red (unhelpful)
        
        # Depth 2 (From N1_1)
        'N2_1': (1, 2.5, 'Step 2:\nBooks = $20. Total = $26\nEval: likely', '#a6e3a1'),
        'N2_2': (3, 2.5, 'Step 2:\nChange = $50 - $6 = $44\nEval: unlikely (forgot books)', '#f38ba8'),
        
        # Depth 2 (From N1_2)
        'N2_3': (5, 2.5, 'Step 2:\nPens cost $5? Total = $35\nEval: impossible (hallucination)', '#f38ba8'),
        
        # Depth 3 (From N2_1)
        'N3_1': (1, 1.5, 'Step 3:\nChange: $50 - $26 = $24\nEval: SUCCESS!', '#a6e3a1'),
        'N3_2': (5, 1.5, 'Step 3:\nChange: $50 + $26 = $76\nEval: impossible (math error)', '#f38ba8')
    }

    edges = [
        ('Root', 'N1_1'), ('Root', 'N1_2'), ('Root', 'N1_3'),
        ('N1_1', 'N2_1'), ('N1_1', 'N2_2'),
        ('N1_2', 'N2_3'),
        ('N2_1', 'N3_1'),
        ('N2_1', 'N3_2')
    ]

    G.add_edges_from(edges)
    pos = {node: (x, y) for node, (x, y, _, _) in nodes.items()}

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', 
                           arrowsize=20, edge_color='#888888', width=2, connectionstyle='arc3,rad=0.0')

    # Draw nodes as rounded bounding boxes
    for node, (x, y, label, color) in nodes.items():
        # Draw background box
        box_width = 2.4 if 'Input' in label else 2.1
        box = patches.FancyBboxPatch((x - box_width/2, y - 0.4), box_width, 0.8, 
                                     boxstyle="round,pad=0.1", 
                                     ec="#333333", fc=color, lw=1.5, alpha=0.9, zorder=2)
        ax.add_patch(box)
        
        # Text color
        text_color = 'black'
        font_weight = 'bold' if 'SUCCESS' in label or 'Input' in label else 'normal'
        
        # Larger font size
        ax.text(x, y, label, ha='center', va='center', fontsize=12, 
                color=text_color, fontweight=font_weight, zorder=3, family='sans-serif')

    # Styling
    ax.axis('off')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0.5, 5.5)

    # Legend
    legend_elements = [
        patches.Patch(facecolor='#a6e3a1', edgecolor='#333333', label='Valid Logic (Likely)'),
        patches.Patch(facecolor='#f9e2af', edgecolor='#333333', label='Alternative Path (Likely)'),
        patches.Patch(facecolor='#f38ba8', edgecolor='#333333', label='Hallucination / Error (Impossible)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False, 
              labelcolor='black', fontsize=14)

    plt.tight_layout()
    
    output_filename = 'gsm8k_search_tree.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Plot saved to: {output_filename}")

if __name__ == '__main__':
    create_gsm8k_tree_plot()
