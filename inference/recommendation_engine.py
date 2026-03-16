"""
recommendation_engine.py — Industrial Action Logic
===================================================
Translates Fibrillation Index (FI) and length metrics into 
actionable recommendations for a paper mill operator.
"""

from typing import List, Dict

def generate_recommendations(metrics_list: List[Dict]) -> str:
    """
    Analyzes the metrics of all detected fibrils in an image and
    returns a formatted string with industrial recommendations.
    
    Args:
        metrics_list: List of dictionaries containing fibril metrics.
        
    Returns:
        Formatted recommendation string.
    """
    if not metrics_list:
        return "⚠️ No fibrils detected. Ensure the image is focused and contains pulp."

    # Compute averages
    avg_fi = sum(m.get("fibrillation_index", 0) for m in metrics_list) / len(metrics_list)
    avg_len = sum(m.get("total_length_um", 0) for m in metrics_list) / len(metrics_list)
    
    # Analyze the Fibrillation Index (FI)
    fi_status = ""
    fi_recommendation = ""
    color = ""
    
    if avg_fi < 1.0:
        fi_status = "Severely Under-Refined"
        fi_recommendation = "INCREASE REFINING. The fibers are too smooth and will not interlock well, leading to weak paper."
        color = "🔴"
    elif 1.0 <= avg_fi < 2.0:
        fi_status = "Under-Refined"
        fi_recommendation = "CONTINUE REFINING. The pulp needs more fibrillation to reach optimal bonding strength."
        color = "🟠"
    elif 2.0 <= avg_fi <= 3.5:
        fi_status = "Optimal Fibrillation"
        fi_recommendation = "STOP REFINING. The pulp is in perfect condition for strong paper. Further refining will waste energy."
        color = "🟢"
    else:
        fi_status = "Over-Refined"
        fi_recommendation = "WARNING: STOP IMMEDIATELY. Pulp is heavily over-refined. Energy is being wasted and fibers may be damaged."
        color = "🔴"
        
    # Analyze Length Preservation (Are we cutting the fibers too short?)
    length_warning = ""
    if avg_len < 30.0:
        length_warning = "\n⚠️ LENGTH WARNING: Average fiber length is suspiciously short (<30μm). The refiner plates may be cutting the fibers instead of brushing them. Check plate clearance."
        
    # Format the output report
    report = f"""
======================================================
🏭 INDUSTRIAL PULP REFINING RECOMMENDATION
======================================================
Batch Analysis (Based on {len(metrics_list)} observed fibrils)

▶ Average Fibrillation Index (FI): {avg_fi:.2f}
▶ Current Status: {color} {fi_status}

💡 ACTIONABLE RECOMMENDATION:
{fi_recommendation}
{length_warning}
======================================================
"""
    return report

def print_recommendation(metrics_list: List[Dict]):
    """Helper function to directly print the generated recommendation."""
    print(generate_recommendations(metrics_list))
