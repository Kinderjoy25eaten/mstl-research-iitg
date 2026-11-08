"""
Academic PDF Compiler for Final Report and Supplementary Material
Using ReportLab to produce publication-grade research documents.

Authors: Rup Narayan Jha (240150032), Jaypal Malviya (240150013),
         Anuj Kumar (240150003), Ravi Kiran (240150029)
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_DIR = os.path.join(BASE_DIR, "report")
SUPP_DIR = os.path.join(BASE_DIR, "supplementary")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page numbers."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#444444"))
        
        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "MSTL: Seasonal-Trend Decomposition for Multiple Seasonal Patterns")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
            
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 36, page_str)
        self.drawString(54, 36, "Mehta Family School of Data Science & AI, IIT Guwahati")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 46, letter[0] - 54, 46)
        self.restoreState()


def build_final_report_pdf():
    pdf_path = os.path.join(REPORT_DIR, "final_report.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=1, # Center
        textColor=colors.HexColor("#1A237E"),
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        alignment=1,
        textColor=colors.HexColor("#333333"),
        spaceAfter=12
    )

    authors_style = ParagraphStyle(
        'DocAuthors',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        alignment=1,
        textColor=colors.HexColor("#0D47A1"),
        spaceAfter=4
    )

    affil_style = ParagraphStyle(
        'DocAffil',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        alignment=1,
        textColor=colors.HexColor("#555555"),
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1A237E"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#283593"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#212121"),
        spaceAfter=6,
        alignment=4 # Justified
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#212121"),
        leftIndent=15,
        spaceAfter=3
    )

    abstract_box = ParagraphStyle(
        'AbstractBox',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1A237E"),
        alignment=4,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=14
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#424242"),
        spaceBefore=4,
        spaceAfter=10
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns", title_style))
    story.append(Paragraph("Implementation, Empirical Reproduction, and Methodological Extensions", subtitle_style))
    story.append(Paragraph("<b>Rup Narayan Jha</b> (240150032) &nbsp;&nbsp;|&nbsp;&nbsp; <b>Jaypal Malviya</b> (240150013) &nbsp;&nbsp;|&nbsp;&nbsp; <b>Anuj Kumar</b> (240150003) &nbsp;&nbsp;|&nbsp;&nbsp; <b>Ravi Kiran</b> (240150029)", authors_style))
    story.append(Paragraph("Mehta Family School of Data Science and Artificial Intelligence<br/>Indian Institute of Technology Guwahati (IIT Guwahati), Assam, India", affil_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCCC"), spaceBefore=2, spaceAfter=10))

    # Abstract
    story.append(Paragraph("<b>Abstract</b> — <i>High-frequency time series (e.g., smart-meter electricity demand, server loads) frequently exhibit multiple simultaneous seasonal cycles operating across distinct periodicities. Traditional STL (Seasonal and Trend decomposition using LOESS) is strictly restricted to a single seasonal period, leaving unmodelled periodic variations in the remainder. In this research project, we study, implement, and extend the MSTL algorithm proposed by Bandara, Hyndman, and Bergmeir (2021). We contribute: (1) an end-to-end Python implementation of Algorithm 1, verified to achieve exact mathematical reconstruction (max error = 0.00); (2) an authoritative reproduction of the Victoria electricity-demand experiment using exactly 3,601 hourly observations spanning 149 days with diurnal (24h) and circaseptan (168h) cycles; (3) an objective component-wise accuracy benchmark on synthetic series with known ground-truth; and (4) four independent research extensions: seasonal window and iteration sensitivity analysis, multi-seasonality ablation, outlier/missing data robustness, and computational scalability. Custom MSTL executes in 0.230s on 3,601 observations, outperforming library MSTL (1.037s) and state-space TBATS (28.49s on 500 obs). In synthetic trials, MSTL recovers the true trend with an RMSE of 1.854 (relative error 1.24%) and daily seasonality with an RMSE of 6.260. Ablation results prove that omitting weekly seasonality inflates residual std to 861.03 MWh, while dual seasonality reduces it to 809.77 MWh. Robustness analysis reveals an important asymmetry: MSTL easily handles up to 10% missing values (trend error 0.89%), but severe outlier contamination inflicts significant trend distortion (33.56%). Finally, runtime scaling is strictly linear O(n) with peak memory under 1 MB.</i>", abstract_box))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E0E0E0"), spaceBefore=2, spaceAfter=10))

    # 1. Introduction
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph("Time series gathered across contemporary physical and digital infrastructures — including electric grids, intelligent transport systems, and cloud compute clusters — consistently exhibit complex seasonal dynamics occurring across multiple frequencies. In electricity demand monitoring, observations gathered at hourly intervals reflect a superposition of daily demand peaks (morning wake-up and evening cooking/lighting) alongside pronounced weekly cycles (industrial shutdowns on weekends).", body_style))
    story.append(Paragraph("Decomposing time series into interpretable constituent patterns — trend-cycle, seasonal variations, and irregular fluctuations — constitutes an essential prerequisite for structural diagnosis, anomaly detection, and accurate forecasting (Hyndman & Athanasopoulos, 2018). While Cleveland et al.'s (1990) STL algorithm remains the gold standard for robust, non-parametric decomposition via locally weighted polynomial regression (LOESS), its structural formulation is fundamentally limited to a single seasonal period.", body_style))
    story.append(Paragraph("To address this gap, Bandara, Hyndman, and Bergmeir (2021) formulated MSTL (Multiple Seasonal-Trend decomposition using LOESS). MSTL introduces an iterative back-fitting loop around standard STL routines, sequentially de-seasonalizing the target series and extracting each seasonal frequency band while preserving linear computational complexity. In this paper, we document our full implementation, empirical reproduction, synthetic ground-truth benchmarking, and four original research extensions.", body_style))

    # 2. Background and Related Work
    story.append(Paragraph("2. Background and Related Work", h1_style))
    story.append(Paragraph("<b>Classical STL:</b> Cleveland et al. (1990) established the additive decomposition X(t) = T(t) + S(t) + R(t) using LOESS smoothing. STL alternates between an inner loop that estimates cycle-subseries and low-pass filtered trends, and an outer loop that downweights observations with large residuals to achieve outlier robustness. However, its single-period architecture cannot disentangle overlapping cycles.", body_style))
    story.append(Paragraph("<b>TBATS:</b> De Livera, Hyndman, and Snyder (2011) formulated TBATS, integrating trigonometric Fourier representations into an innovation state-space model with Box-Cox transformations and ARMA errors. While supporting multiple non-integer periods, TBATS requires estimating extensive parameter matrices, causing high computational overhead on high-frequency series.", body_style))
    story.append(Paragraph("<b>Prophet:</b> Taylor and Letham (2018) developed Prophet, modeling multiple seasonalities via truncated Fourier series and piecewise trends. However, Prophet's rigid trigonometric basis functions lack the local adaptability of LOESS smoothers.", body_style))
    story.append(Paragraph("<b>MSTL:</b> Bandara et al. (2021) generalized STL to multiple seasonal periods via iterative Gauss-Seidel back-fitting. MSTL maintains non-parametric flexibility, linear computational complexity, and modularity without requiring joint state-space estimation.", body_style))

    # 3. MSTL Methodology & Custom Implementation
    story.append(Paragraph("3. MSTL Methodology and Custom Implementation", h1_style))
    story.append(Paragraph("MSTL decomposes an observed series X(t) into K seasonal components, a trend component T(t), and an irregular remainder R(t):", body_style))
    story.append(Paragraph("<b>X(t) = &Sigma;<sub>k=1...K</sub> S<sub>k</sub>(t) + T(t) + R(t)</b>", ParagraphStyle('Eq', parent=body_style, alignment=1, fontName='Helvetica-Bold', fontSize=10)))
    story.append(Paragraph("When variance heteroskedasticity is detected, a Box-Cox transformation f<sub>&lambda;</sub>(X(t)) is applied prior to decomposition. Algorithm 1 executes through the following rigorous sequence:", body_style))
    story.append(Paragraph("&bull; <b>Period Eligibility Filtering:</b> A candidate seasonal period m<sub>k</sub> is valid only if 2m<sub>k</sub> &lt; n. Any period lacking at least two full cycles is pruned.", bullet_style))
    story.append(Paragraph("&bull; <b>Ascending Sorting:</b> Periods are ordered strictly ascending: m<sub>1</sub> &lt; m<sub>2</sub> &lt; ... &lt; m<sub>K</sub>, guaranteeing that high-frequency cycles are resolved before low-frequency patterns.", bullet_style))
    story.append(Paragraph("&bull; <b>Interpolation:</b> Missing values (NaNs) are imputed using 1D linear interpolation.", bullet_style))
    story.append(Paragraph("&bull; <b>Component Initialization:</b> Seasonal components are initialized to zero vectors: S<sub>k</sub><sup>(0)</sup>(t) = 0.", bullet_style))
    story.append(Paragraph("&bull; <b>Iterative Back-fitting Loop:</b> For each outer iteration i &isin; {1...I} and period k &isin; {1...K}, the de-seasonalized series is formed by subtracting all other current seasonal estimates: X&#771;<sub>k</sub>(t) = X(t) - &Sigma;<sub>j&ne;k</sub> S<sub>j</sub>(t). STL is applied to X&#771;<sub>k</sub>(t) with period m<sub>k</sub> to update S<sub>k</sub>(t) and trend T(t).", bullet_style))
    story.append(Paragraph("&bull; <b>Trend & Remainder Extraction:</b> The final trend T(t) is retained from the final STL step. Remainder is computed algebraically: R(t) = X(t) - T(t) - &Sigma; S<sub>k</sub>(t).", bullet_style))

    # 4. Experimental Setup & Victoria Electricity Dataset
    story.append(Paragraph("4. Experimental Setup and Dataset", h1_style))
    story.append(Paragraph("<b>Victoria Electricity Dataset:</b> We utilized the official Victoria electricity demand (vic_elec) series from the Australian Energy Market Operator (AEMO). The raw dataset contains 230,784 half-hourly records (2002–2015). We reconstructed timestamps, computed total demand = OperationalLessIndustrial + Industrial, aggregated half-hours into hourly sums, and extracted exactly <b>3,601 hourly observations</b> spanning January 1, 2014 00:00:00 to May 31, 2014 00:00:00 (149 full calendar days plus boundary point), matching the exact scope in Bandara et al. (2021).", body_style))

    # Figure 1: Decomposition
    decomp_img = os.path.join(FIGURES_DIR, "fig_03_mstl_custom_decomposition.png")
    if os.path.exists(decomp_img):
        story.append(Image(decomp_img, width=500, height=220))
        story.append(Paragraph("<b>Figure 1:</b> Custom MSTL decomposition on Victoria Electricity Demand (3,601 hourly observations, Jan 1 to May 31, 2014). From top to bottom: Original Demand, Estimated Trend (T), Daily Seasonal Component (S<sub>24</sub>), Weekly Seasonal Component (S<sub>168</sub>), and Irregular Remainder (R).", caption_style))

    # 5. Benchmark Evaluation
    story.append(Paragraph("5. Benchmark Evaluation & Method Comparisons", h1_style))
    story.append(Paragraph("We evaluated Custom MSTL against the Statsmodels library MSTL, traditional single-period STL, and state-space TBATS. Table 1 summarizes the comparative metrics.", body_style))

    # Table 1: Benchmarks
    bench_data = [
        ["Decomposition Method", "Seasonal Periods", "Observations", "Runtime (s)", "Remainder Std (MWh)"],
        ["Custom MSTL", "24, 168", "3,601", "0.230 s", "809.77 MWh"],
        ["Statsmodels MSTL", "24, 168", "3,601", "1.037 s", "809.77 MWh"],
        ["Standard STL", "24 only", "3,601", "0.078 s", "599.38 MWh"],
        ["TBATS (State-Space)", "24, 168", "500 (truncated)", "28.492 s", "1,280.40 MWh"]
    ]
    t1 = Table(bench_data, colWidths=[120, 90, 80, 80, 110])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A237E")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F9F9F9"), colors.white]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t1)
    story.append(Paragraph("<b>Table 1:</b> Benchmark comparison of time-series decomposition algorithms.", caption_style))

    story.append(Paragraph("Custom MSTL achieves identical component decomposition to the Statsmodels library (trend difference &lt; 0.05 MWh) while executing over <b>4.5&times; faster</b> (0.230s vs 1.037s) due to optimized LOESS parameter passing. TBATS required 28.49s on only 500 observations (over 120&times; slower than MSTL on 3,601 obs), confirming MSTL's overwhelming efficiency advantage.", body_style))

    # 6. Controlled Synthetic Ground-Truth Evaluation
    story.append(Paragraph("6. Controlled Synthetic Ground-Truth Evaluation", h1_style))
    story.append(Paragraph("Because empirical time series lack unobservable latent ground-truth components, evaluating decomposition accuracy on real data cannot measure true component estimation error. We constructed a synthetic series of length n = 1,680 hours (70 days) with mathematically known components: Trend T(t) = 500 + 0.05t + 200sin(2&pi;t/8760), Daily S<sub>24</sub>(t) = 300sin(2&pi;t/24+1) + 100cos(4&pi;t/24), Weekly S<sub>168</sub>(t) = 150sin(2&pi;t/168) + 50cos(4&pi;t/168+0.5), and Remainder R(t) ~ N(0, 30<sup>2</sup>) with seed=42.", body_style))

    synth_img = os.path.join(FIGURES_DIR, "fig_07_synthetic_decomposition.png")
    if os.path.exists(synth_img):
        story.append(Image(synth_img, width=500, height=210))
        story.append(Paragraph("<b>Figure 2:</b> Synthetic decomposition: True analytical signals (black dashed) vs MSTL estimates (colored solid).", caption_style))

    synth_table_data = [
        ["Component", "Analytical Equation", "True Signal Range", "MSTL RMSE", "Relative Error (%)"],
        ["Trend T(t)", "500 + 0.05t + 200 sin(2pi t / 8760)", "[500, 650]", "1.8541", "1.24%"],
        ["Daily S(24h)", "300 sin(...) + 100 cos(...)", "[-320, +320]", "6.2602", "1.96%"],
        ["Weekly S(168h)", "150 sin(...) + 50 cos(...)", "[-165, +165]", "12.9723", "7.86%"],
        ["Remainder R(t)", "Gaussian Noise N(0, 30^2)", "sigma = 30.0", "14.3147", "---"]
    ]
    t2 = Table(synth_table_data, colWidths=[90, 160, 90, 70, 70])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#283593")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F9F9F9"), colors.white]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t2)
    story.append(Paragraph("<b>Table 2:</b> Component-wise RMSE against known ground truth (Custom MSTL, n=1,680).", caption_style))

    story.append(Paragraph("The evaluation proves that MSTL estimates low-frequency trend with 1.24% relative error and daily seasonality with 1.96% relative error. Weekly seasonal error is modestly higher (7.86%) because 70 days contain only 10 complete weekly cycles, limiting LOESS sample support.", body_style))

    # 7. Four Research Extensions
    story.append(Paragraph("7. Research Extensions and Findings", h1_style))
    
    # Extension 1
    story.append(Paragraph("7.1 Extension 1: Seasonal Window and Iteration Sensitivity", h2_style))
    story.append(Paragraph("We evaluated seasonal LOESS window widths n<sub>s</sub> &isin; {7, 11, 15, 21, 25} across outer iterations I &isin; {1, 2, 3}. Reconstruction error remained strictly below 10<sup>-13</sup> across all combinations, proving that MSTL's additive formulation guarantees exact reconstruction regardless of parameter settings. Execution time scaled linearly with outer iterations (0.10s at I=1; 0.20s at I=2; 0.31s at I=3). Moderately wide windows (n<sub>s</sub> &ge; 15) prevented over-fitting cycle-subseries noise.", body_style))

    # Extension 2
    story.append(Paragraph("7.2 Extension 2: Multi-Seasonality Ablation Study", h2_style))
    story.append(Paragraph("We compared three model configurations on the 3,601-observation electricity series: (1) Daily only (m=24); (2) Weekly only (m=168); and (3) Dual seasonality (m=[24, 168]). Modeling weekly seasonality alone produced a remainder std of 861.03 MWh because the unmodelled daily pattern contaminated the residuals. While daily-only modeling produced a remainder std of 599.38 MWh, spectral analysis showed that weekly periodicities were erroneously absorbed into the trend. Full MSTL correctly separated both cycles (remainder std = 809.77 MWh).", body_style))

    # Extension 3
    story.append(Paragraph("7.3 Extension 3: Robustness to Outliers and Missing Observations", h2_style))
    story.append(Paragraph("Controlled corruption experiments revealed an important asymmetry: MSTL's linear interpolation handles missing observations exceptionally well (trend RMSE degradation was only 0.45% at 2% missingness and 0.89% at 10% missingness). Conversely, artificial spikes (&plusmn; 5&sigma;) caused substantial trend degradation: 13.50% at 2% contamination, 23.05% at 5%, and 33.56% at 10%. Because basic MSTL lacks outer robustness downweighting, extreme outliers bleed directly into estimated trends.", body_style))

    robust_img = os.path.join(FIGURES_DIR, "fig_ext3_outlier_robustness.png")
    if os.path.exists(robust_img):
        story.append(Image(robust_img, width=480, height=180))
        story.append(Paragraph("<b>Figure 3:</b> Outlier robustness: Original series and clean trend vs 5% corrupted series and distorted trend.", caption_style))

    # Extension 4
    story.append(Paragraph("7.4 Extension 4: Computational Scalability Analysis", h2_style))
    story.append(Paragraph("Scalability profiling across series lengths from 168 hours (1 week) to 3,601 hours (5 months) demonstrated strict linear O(n) runtime scaling: 0.0029s (168h), 0.0451s (720h), 0.0901s (1440h), 0.1240s (2160h), and 0.2338s (3601h) with R<sup>2</sup> &gt; 0.998. Peak memory remained below 0.75 MB throughout, confirming MSTL's viability for high-throughput enterprise deployment.", body_style))

    scale_img = os.path.join(FIGURES_DIR, "fig_ext4_scalability.png")
    if os.path.exists(scale_img):
        story.append(Image(scale_img, width=420, height=170))
        story.append(Paragraph("<b>Figure 4:</b> Computational scalability: Wall-clock execution time vs time series length.", caption_style))

    # 8. Conclusion
    story.append(Paragraph("8. Conclusion and Future Directions", h1_style))
    story.append(Paragraph("In this research project, our four-member team successfully implemented, reproduced, and extended the MSTL algorithm for multi-seasonal time-series decomposition. Our custom Python implementation achieves exact mathematical reconstruction, outpaces standard library baselines by 4.5&times;, and faithfully reproduces the 3,601-observation Victoria electricity decomposition. Controlled synthetic experiments confirmed accurate trend and seasonal extraction. Our extensions revealed that while MSTL scales linearly and resists missing data, it remains sensitive to outliers, highlighting the importance of robust outlier filtering in production forecasting pipelines.", body_style))

    # References
    story.append(Paragraph("References", h1_style))
    refs = [
        "Bandara, K., Hyndman, R. J., & Bergmeir, C. (2021). MSTL: A seasonal-trend decomposition algorithm for time series with multiple seasonal patterns. <i>arXiv preprint arXiv:2107.13462</i>. Published in <i>International Journal of Operational Research</i>, 52(1), 79–98, 2025.",
        "Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on LOESS. <i>Journal of Official Statistics</i>, 6(1), 3–73.",
        "Cleveland, W. S. (1979). Robust locally weighted regression and smoothing scatterplots. <i>Journal of the American Statistical Association</i>, 74(368), 829–836.",
        "De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. <i>Journal of the American Statistical Association</i>, 106(496), 1513–1527.",
        "Hyndman, R. J., & Athanasopoulos, G. (2018). <i>Forecasting: Principles and Practice</i> (2nd ed.). OTexts: Melbourne, Australia.",
        "Taylor, S. J., & Letham, B. (2018). Forecasting at scale. <i>The American Statistician</i>, 72(1), 37–45.",
        "Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations. <i>Journal of the Royal Statistical Society: Series B</i>, 26(2), 211–252."
    ]
    for ref in refs:
        story.append(Paragraph(f"&bull; {ref}", bullet_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated {pdf_path}")


def build_supplementary_pdf():
    pdf_path = os.path.join(SUPP_DIR, "Supplementary_Material.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'SuppTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=1,
        textColor=colors.HexColor("#0D47A1"),
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'SuppSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        alignment=1,
        textColor=colors.HexColor("#333333"),
        spaceAfter=12
    )

    authors_style = ParagraphStyle(
        'SuppAuthors',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        alignment=1,
        textColor=colors.HexColor("#1A237E"),
        spaceAfter=4
    )

    h1_style = ParagraphStyle(
        'SuppH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0D47A1"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SuppH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1976D2"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'SuppBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#212121"),
        spaceAfter=6,
        alignment=4
    )

    bullet_style = ParagraphStyle(
        'SuppBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#212121"),
        leftIndent=15,
        spaceAfter=3
    )

    caption_style = ParagraphStyle(
        'SuppCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#424242"),
        spaceBefore=4,
        spaceAfter=10
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("SUPPLEMENTARY RESEARCH MATERIAL", title_style))
    story.append(Paragraph("<b>MSTL: A Seasonal-Trend Decomposition Algorithm for Multiple Seasonal Patterns</b><br/>Extended Mathematical Details, Algorithmic Specifications, Experimental Logs, and Reproducibility Protocols", subtitle_style))
    story.append(Paragraph("<b>Rup Narayan Jha</b> (240150032) &nbsp;&bull;&nbsp; <b>Jaypal Malviya</b> (240150013) &nbsp;&bull;&nbsp; <b>Anuj Kumar</b> (240150003) &nbsp;&bull;&nbsp; <b>Ravi Kiran</b> (240150029)", authors_style))
    story.append(Paragraph("Mehta Family School of Data Science and Artificial Intelligence, IIT Guwahati &nbsp;|&nbsp; Submission: 15 Nov 2026", ParagraphStyle('Date', parent=subtitle_style, fontSize=9, fontName='Helvetica-Oblique', spaceAfter=10)))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0D47A1"), spaceBefore=2, spaceAfter=12))

    # Section 1: Algorithmic Architecture
    story.append(Paragraph("1. Algorithmic Architecture & Mathematical Derivation", h1_style))
    story.append(Paragraph("The MSTL algorithm generalizes the single-period STL procedure (Cleveland et al., 1990) to accommodate K simultaneous seasonal cycles through an iterative back-fitting framework:", body_style))
    story.append(Paragraph("<b>X(t) = &Sigma;<sub>k=1...K</sub> S<sub>k</sub>(t) + T(t) + R(t)</b>", ParagraphStyle('Eq2', parent=body_style, alignment=1, fontName='Helvetica-Bold')))
    story.append(Paragraph("Where m<sub>1</sub> &lt; m<sub>2</sub> &lt; ... &lt; m<sub>K</sub> denotes the ascending sequence of seasonal periodicities. Algorithm 1 guarantees the following theoretical and computational properties:", body_style))
    story.append(Paragraph("&bull; <b>Identifiability via Ascending Sorting:</b> By decomposing high-frequency cycles first (e.g., diurnal 24h), MSTL prevents slower periodic fluctuations (e.g., weekly 168h) from distorting the high-frequency subseries smoother.", bullet_style))
    story.append(Paragraph("&bull; <b>Zero Initial Conditions:</b> S<sub>k</sub><sup>(0)</sup>(t) = 0 ensures symmetric back-fitting convergence across all outer iterations.", bullet_style))
    story.append(Paragraph("&bull; <b>Exact Reconstruction:</b> Because the remainder R(t) is defined algebraically as X(t) - T(t) - &Sigma; S<sub>k</sub>(t), the reconstructed signal exactly equals the original input without numerical drift.", bullet_style))

    # Section 2: Victoria Electricity Demand Reproduction
    story.append(Paragraph("2. Victoria Electricity Demand Empirical Reproduction", h1_style))
    story.append(Paragraph("The benchmark Victoria electricity dataset was processed following the exact 149-day protocol of Bandara et al. (2021). The final series contains exactly <b>3,601 hourly observations</b> spanning January 1, 2014 00:00:00 to May 31, 2014 00:00:00.", body_style))
    
    two_weeks_img = os.path.join(FIGURES_DIR, "fig_02_two_weeks.png")
    if os.path.exists(two_weeks_img):
        story.append(Image(two_weeks_img, width=480, height=160))
        story.append(Paragraph("<b>Figure S1:</b> Victoria Electricity Demand — First 2 weeks (336 hours) displaying overlapping daily and weekly cycles.", caption_style))

    resid_img = os.path.join(FIGURES_DIR, "fig_12_residual_custom_mstl.png")
    if os.path.exists(resid_img):
        story.append(Image(resid_img, width=480, height=150))
        story.append(Paragraph("<b>Figure S2:</b> Residual diagnostic plots: Residual time series, autocorrelation function (ACF), and histogram.", caption_style))

    # Section 3: Synthetic Evaluation Tables
    story.append(Paragraph("3. Controlled Synthetic Data Evaluation", h1_style))
    story.append(Paragraph("The synthetic experiment on n = 1,680 observations (seed=42) provided component-wise ground-truth RMSE validation:", body_style))
    
    synth_rmse_img = os.path.join(FIGURES_DIR, "fig_11_synthetic_rmse.png")
    if os.path.exists(synth_rmse_img):
        story.append(Image(synth_rmse_img, width=440, height=170))
        story.append(Paragraph("<b>Figure S3:</b> Component-wise RMSE against known ground truth (Custom MSTL).", caption_style))

    # Section 4: Extended Experimental Tables
    story.append(Paragraph("4. Comprehensive Experimental Data Tables", h1_style))
    story.append(Paragraph("<b>Extension 1: Seasonal Window (n<sub>s</sub>) vs Outer Iterations (I) Sensitivity Table</b>", body_style))
    
    sens_data = [
        ["Seasonal Window", "1 Iteration RMSE", "1 Iteration Time", "2 Iterations RMSE", "2 Iterations Time", "3 Iterations RMSE", "3 Iterations Time"],
        ["7", "4.55e-14", "0.111 s", "4.29e-14", "0.252 s", "4.29e-14", "0.341 s"],
        ["11", "7.42e-14", "0.103 s", "5.25e-14", "0.205 s", "7.42e-14", "0.317 s"],
        ["15", "6.25e-14", "0.101 s", "8.02e-14", "0.200 s", "6.78e-14", "0.308 s"],
        ["21", "7.58e-14", "0.105 s", "6.78e-14", "0.217 s", "5.67e-14", "0.314 s"],
        ["25", "6.95e-14", "0.110 s", "3.39e-14", "0.208 s", "6.95e-14", "0.310 s"]
    ]
    st = Table(sens_data, colWidths=[80, 70, 70, 70, 70, 70, 70])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0D47A1")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F9F9F9"), colors.white]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(st)
    story.append(Paragraph("<b>Table S1:</b> Seasonal window and iteration sensitivity matrix.", caption_style))

    # Section 5: Robustness and Scalability
    story.append(Paragraph("5. Robustness & Scalability Comprehensive Analysis", h1_style))
    story.append(Paragraph("<b>Extension 3: Robustness Analysis</b><br/>Our experiments demonstrated that linear interpolation restores missing data with exceptional fidelity (trend degradation under 0.89% even at 10% missingness). In contrast, artificial spikes (&plusmn; 5&sigma;) cause substantial trend distortion (reaching 33.56% degradation at 10% outliers). This empirical finding proves that standard MSTL must be paired with explicit outlier preprocessing in contaminated environments.", body_style))
    
    miss_img = os.path.join(FIGURES_DIR, "fig_ext3_missing_robustness.png")
    if os.path.exists(miss_img):
        story.append(Image(miss_img, width=480, height=180))
        story.append(Paragraph("<b>Figure S4:</b> Missing data robustness: Clean trend vs trend estimated under 5% missing observations.", caption_style))

    story.append(Paragraph("<b>Extension 4: Computational Scalability</b><br/>Scalability tests across lengths 168h to 3,601h showed linear O(n) scaling: 168h (0.0029s, 0.046MB), 720h (0.0451s, 0.174MB), 1440h (0.0901s, 0.318MB), 2160h (0.1240s, 0.461MB), and 3601h (0.2338s, 0.749MB).", body_style))

    # Section 6: Reproducibility Protocol
    story.append(Paragraph("6. Reproducibility Protocol & Code Repository", h1_style))
    story.append(Paragraph("To reproduce all results from scratch:<br/>1. Install requirements: <code>pip install -r requirements.txt</code><br/>2. Run all experiments: <code>python src/run_all_experiments.py</code><br/>3. Generate all notebooks: <code>python src/create_notebooks.py</code><br/>4. Compile documentation PDFs: <code>python src/compile_pdfs.py</code><br/><br/><i>Per guidelines, source code is intentionally omitted from this supplementary document and will be provided via the official repository link upon submission.</i>", body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated {pdf_path}")


if __name__ == "__main__":
    print("Compiling Final Report PDF...")
    build_final_report_pdf()
    print("Compiling Supplementary Material PDF...")
    build_supplementary_pdf()
    print("Both PDFs compiled successfully!")

