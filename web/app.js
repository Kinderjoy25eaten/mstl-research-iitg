/**
 * MSTL Interactive Live Decomposition Explorer
 * Authors: Rup Narayan Jha, Jaypal Malviya, Anuj Kumar, Ravi Kiran
 * Mehta Family School of Data Science & AI, IIT Guwahati
 */

let vicData = null;
let synthData = null;
let currentDataset = 'vic';

const visibleComponents = {
  original: true,
  trend: true,
  daily: true,
  weekly: true,
  remainder: true
};

const colors = {
  original: '#3b82f6',  // Blue
  trend: '#ef4444',     // Red/Crimson
  daily: '#10b981',     // Green
  weekly: '#f59e0b',    // Orange
  remainder: '#8b5cf6'  // Purple
};

async function loadData() {
  try {
    const [vicRes, synthRes] = await Promise.all([
      fetch('data/vic_elec_decomposition.json'),
      fetch('data/synthetic_decomposition.json')
    ]);
    
    vicData = await vicRes.json();
    synthData = await synthRes.json();
    
    renderChart();
  } catch (error) {
    console.error('Error loading data:', error);
    document.getElementById('plotlyChart').innerHTML = `
      <div style="text-align: center; padding-top: 200px; color: #ef4444;">
        <i class="fa-solid fa-triangle-exclamation fa-2x"></i>
        <p style="margin-top: 10px; font-weight: 600;">Failed to load decomposition datasets.</p>
        <p style="font-size: 0.85rem; color: #64748b;">Please ensure web/data JSON files are accessible.</p>
      </div>
    `;
  }
}

function renderChart() {
  const chartEl = document.getElementById('plotlyChart');
  if (!chartEl) return;

  const traces = [];

  if (currentDataset === 'vic' && vicData) {
    const x = vicData.timestamps;

    if (visibleComponents.original) {
      traces.push({
        x: x,
        y: vicData.original,
        type: 'scatter',
        mode: 'lines',
        name: 'Original Demand X(t)',
        line: { color: colors.original, width: 1.2 }
      });
    }

    if (visibleComponents.trend) {
      traces.push({
        x: x,
        y: vicData.trend,
        type: 'scatter',
        mode: 'lines',
        name: 'Trend T(t)',
        line: { color: colors.trend, width: 2 }
      });
    }

    if (visibleComponents.daily) {
      traces.push({
        x: x,
        y: vicData.seasonal_24,
        type: 'scatter',
        mode: 'lines',
        name: 'Daily Seasonal S_24(t)',
        line: { color: colors.daily, width: 1 }
      });
    }

    if (visibleComponents.weekly) {
      traces.push({
        x: x,
        y: vicData.seasonal_168,
        type: 'scatter',
        mode: 'lines',
        name: 'Weekly Seasonal S_168(t)',
        line: { color: colors.weekly, width: 1.2 }
      });
    }

    if (visibleComponents.remainder) {
      traces.push({
        x: x,
        y: vicData.remainder,
        type: 'scatter',
        mode: 'lines',
        name: 'Remainder R(t)',
        line: { color: colors.remainder, width: 0.8 },
        opacity: 0.7
      });
    }

    const layout = {
      title: {
        text: 'Victoria Electricity Demand — MSTL Decomposition (3,601 Observations)',
        font: { family: 'Inter', size: 14, color: '#1e293b' }
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: '#ffffff',
      margin: { l: 60, r: 20, t: 40, b: 50 },
      xaxis: {
        title: 'Time',
        rangeslider: { visible: true },
        gridcolor: '#f1f5f9'
      },
      yaxis: {
        title: 'Demand (MWh)',
        gridcolor: '#f1f5f9'
      },
      legend: {
        orientation: 'h',
        y: 1.15,
        x: 0,
        font: { family: 'Inter', size: 10 }
      },
      hovermode: 'x unified'
    };

    Plotly.newPlot(chartEl, traces, layout, { responsive: true, displaylogo: false });

  } else if (currentDataset === 'synth' && synthData) {
    const x = synthData.hours;

    if (visibleComponents.original) {
      traces.push({
        x: x,
        y: synthData.observed,
        type: 'scatter',
        mode: 'lines',
        name: 'Observed X(t)',
        line: { color: colors.original, width: 1 }
      });
    }

    if (visibleComponents.trend) {
      traces.push({
        x: x,
        y: synthData.mstl_trend,
        type: 'scatter',
        mode: 'lines',
        name: 'MSTL Trend (RMSE: 1.85)',
        line: { color: colors.trend, width: 2 }
      });
      traces.push({
        x: x,
        y: synthData.true_trend,
        type: 'scatter',
        mode: 'lines',
        name: 'True Ground Truth Trend',
        line: { color: '#000000', width: 1.5, dash: 'dot' }
      });
    }

    if (visibleComponents.daily) {
      traces.push({
        x: x,
        y: synthData.mstl_daily,
        type: 'scatter',
        mode: 'lines',
        name: 'MSTL Daily S_24 (RMSE: 6.26)',
        line: { color: colors.daily, width: 1 }
      });
    }

    if (visibleComponents.weekly) {
      traces.push({
        x: x,
        y: synthData.mstl_weekly,
        type: 'scatter',
        mode: 'lines',
        name: 'MSTL Weekly S_168 (RMSE: 12.97)',
        line: { color: colors.weekly, width: 1.2 }
      });
    }

    if (visibleComponents.remainder) {
      traces.push({
        x: x,
        y: synthData.mstl_remainder,
        type: 'scatter',
        mode: 'lines',
        name: 'MSTL Remainder R(t)',
        line: { color: colors.remainder, width: 0.8 },
        opacity: 0.7
      });
    }

    const layout = {
      title: {
        text: 'Synthetic Series — Ground Truth vs MSTL Decomposition (1,680 Hours)',
        font: { family: 'Inter', size: 14, color: '#1e293b' }
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: '#ffffff',
      margin: { l: 60, r: 20, t: 40, b: 50 },
      xaxis: {
        title: 'Hour',
        rangeslider: { visible: true },
        gridcolor: '#f1f5f9'
      },
      yaxis: {
        title: 'Component Value',
        gridcolor: '#f1f5f9'
      },
      legend: {
        orientation: 'h',
        y: 1.15,
        x: 0,
        font: { family: 'Inter', size: 10 }
      },
      hovermode: 'x unified'
    };

    Plotly.newPlot(chartEl, traces, layout, { responsive: true, displaylogo: false });
  }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
  loadData();

  // Dataset dropdown
  const datasetSelect = document.getElementById('datasetSelect');
  if (datasetSelect) {
    datasetSelect.addEventListener('change', (e) => {
      currentDataset = e.target.value;
      renderChart();
    });
  }

  // Component toggle buttons
  const toggleButtons = document.querySelectorAll('.toggle-btn');
  toggleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const comp = btn.getAttribute('data-component');
      visibleComponents[comp] = !visibleComponents[comp];
      btn.classList.toggle('active', visibleComponents[comp]);
      renderChart();
    });
  });
});

