/**
 * SpamShield - Interactive Dashboard Charts
 * =========================================
 * Renders Chart.js visualizations for class distribution and model performance.
 */

document.addEventListener("DOMContentLoaded", async () => {
  const distCanvas = document.getElementById("classDistributionChart");
  const perfCanvas = document.getElementById("modelPerformanceChart");

  // Only execute if at least one chart canvas is present on the page
  if (!distCanvas && !perfCanvas) return;

  try {
    const res = await fetch("/api/stats");
    if (!res.ok) {
      console.warn("Could not retrieve /api/stats. Status:", res.status);
      return;
    }

    const data = await res.json();
    if (!data.available) return;

    // -----------------------------------------------------------------------
    // Chart A: Dataset Class Distribution (Doughnut Chart)
    // -----------------------------------------------------------------------
    if (distCanvas && data.class_distribution) {
      new Chart(distCanvas, {
        type: "doughnut",
        data: {
          labels: ["Legitimate (Ham)", "Spam"],
          datasets: [
            {
              data: data.class_distribution.counts,
              backgroundColor: ["#16A34A", "#DC2626"],
              hoverBackgroundColor: ["#15803D", "#B91C1C"],
              borderWidth: 2,
              borderColor: "#FFFFFF"
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: "70%",
          plugins: {
            legend: {
              position: "bottom",
              labels: {
                usePointStyle: true,
                padding: 16,
                font: {
                  family: "Inter",
                  size: 12,
                  weight: 500
                },
                color: "#172033"
              }
            },
            tooltip: {
              backgroundColor: "#14213D",
              titleFont: { family: "Inter", size: 13, weight: 700 },
              bodyFont: { family: "Inter", size: 12 },
              padding: 12,
              cornerRadius: 8,
              callbacks: {
                label: function (context) {
                  const label = context.label || "";
                  const val = context.raw || 0;
                  const total = context.dataset.data.reduce((a, b) => a + b, 0);
                  const pct = total > 0 ? ((val / total) * 100).toFixed(1) : 0;
                  return ` ${label}: ${val.toLocaleString()} (${pct}%)`;
                }
              }
            }
          }
        }
      });
    }

    // -----------------------------------------------------------------------
    // Chart B: Model Performance Metrics (Grouped Bar Chart)
    // -----------------------------------------------------------------------
    if (perfCanvas && data.model_performance) {
      const perf = data.model_performance;

      new Chart(perfCanvas, {
        type: "bar",
        data: {
          labels: ["Ham (Legitimate)", "Spam", "Macro Average"],
          datasets: [
            {
              label: "Precision (%)",
              data: perf.precision,
              backgroundColor: "#2563EB",
              borderRadius: 6
            },
            {
              label: "Recall (%)",
              data: perf.recall,
              backgroundColor: "#16A34A",
              borderRadius: 6
            },
            {
              label: "F1-Score (%)",
              data: perf.f1_score,
              backgroundColor: "#7C3AED",
              borderRadius: 6
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              min: 80,
              max: 100,
              ticks: {
                callback: function (val) {
                  return val + "%";
                },
                font: { family: "Inter", size: 11 },
                color: "#64748B"
              },
              grid: {
                color: "#F1F5F9"
              }
            },
            x: {
              ticks: {
                font: { family: "Inter", size: 11, weight: 600 },
                color: "#172033"
              },
              grid: {
                display: false
              }
            }
          },
          plugins: {
            legend: {
              position: "bottom",
              labels: {
                usePointStyle: true,
                padding: 16,
                font: {
                  family: "Inter",
                  size: 12,
                  weight: 500
                },
                color: "#172033"
              }
            },
            tooltip: {
              backgroundColor: "#14213D",
              titleFont: { family: "Inter", size: 13, weight: 700 },
              bodyFont: { family: "Inter", size: 12 },
              padding: 12,
              cornerRadius: 8,
              callbacks: {
                label: function (context) {
                  return ` ${context.dataset.label}: ${context.raw}%`;
                }
              }
            }
          }
        }
      });
    }
  } catch (err) {
    console.error("Failed to render dashboard charts:", err);
  }
});
