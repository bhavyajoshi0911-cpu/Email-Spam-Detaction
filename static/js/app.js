/**
 * SpamShield - Core Client Application JavaScript
 * ===============================================
 * Handles interactivity, sample presets, AJAX prediction,
 * character counting, modal management, and mobile sidebar navigation.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Mobile Sidebar Toggle
  const sidebar = document.getElementById("appSidebar");
  const sidebarToggle = document.getElementById("sidebarToggle");

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });

    document.addEventListener("click", (e) => {
      if (
        sidebar.classList.contains("open") &&
        !sidebar.contains(e.target) &&
        !sidebarToggle.contains(e.target)
      ) {
        sidebar.classList.remove("open");
      }
    });
  }

  // -------------------------------------------------------------------------
  // Detect Spam Classification Page Features
  // -------------------------------------------------------------------------
  const detectForm = document.getElementById("detectForm");
  const emailSubject = document.getElementById("emailSubject");
  const emailMessage = document.getElementById("emailMessage");
  const counterStats = document.getElementById("counterStats");
  const clearBtn = document.getElementById("clearBtn");
  const errorAlert = document.getElementById("errorAlert");
  const errorMessageText = document.getElementById("errorMessageText");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const btnIcon = document.getElementById("btnIcon");
  const btnSpinner = document.getElementById("btnSpinner");
  const btnLabel = document.getElementById("btnLabel");

  const resultPanel = document.getElementById("resultPanel");
  const verdictIcon = document.getElementById("verdictIcon");
  const verdictHeading = document.getElementById("verdictHeading");
  const verdictExplanation = document.getElementById("verdictExplanation");
  const verdictBadge = document.getElementById("verdictBadge");
  const confidencePercentage = document.getElementById("confidencePercentage");
  const probBarHam = document.getElementById("probBarHam");
  const probBarSpam = document.getElementById("probBarSpam");
  const probHamVal = document.getElementById("probHamVal");
  const probSpamVal = document.getElementById("probSpamVal");
  const riskAssessmentTag = document.getElementById("riskAssessmentTag");
  const analyzedCharCount = document.getElementById("analyzedCharCount");
  const analyzedWordCount = document.getElementById("analyzedWordCount");
  const toggleAnalyzedText = document.getElementById("toggleAnalyzedText");
  const analyzedContentBox = document.getElementById("analyzedContentBox");
  const accordionIcon = document.getElementById("accordionIcon");
  const analyzeAnotherBtn = document.getElementById("analyzeAnotherBtn");

  // Character and Word Counter
  function updateCounters() {
    if (!emailMessage || !counterStats) return;
    const text = emailMessage.value;
    const charCount = text.length;
    const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
    counterStats.textContent = `${charCount.toLocaleString()} characters • ${wordCount.toLocaleString()} words`;
  }

  if (emailMessage) {
    emailMessage.addEventListener("input", updateCounters);
  }

  // Sample Messages
  const samples = {
    sampleLegitMeeting: {
      subject: "Team Sync & Quarterly Project Review",
      message:
        "Hi team,\n\nJust a reminder that our weekly project sync is scheduled for tomorrow at 10:00 AM in Conference Room B. Please review the updated roadmap before the meeting.\n\nBest regards,\nAlex"
    },
    sampleLegitOrder: {
      subject: "Your Order #84920 Confirmed",
      message:
        "Thank you for shopping with CloudStore! Your package has been dispatched and is scheduled to arrive on Thursday. You can track your shipment status at any time inside your member portal.\n\nCustomer Support Team"
    },
    sampleSpamPrize: {
      subject: "CONGRATULATIONS! You won a $1,000 Walmart Gift Card!",
      message:
        "URGENT! You have been selected as our exclusive lucky winner for a complimentary $1,000 gift card! Claim your prize immediately by calling 08712400602 or texting WIN to 88088 now! Terms and conditions apply."
    },
    sampleSpamUrgent: {
      subject: "URGENT: Your account access has been suspended",
      message:
        "SECURITY ALERT: We noticed unauthorized login attempts on your banking account. Your access has been locked to prevent fraud. Click the link below immediately and enter your credentials to verify your identity and restore access: http://verify-secure-bank-login.com/auth"
    }
  };

  Object.keys(samples).forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener("click", () => {
        if (emailSubject) emailSubject.value = samples[id].subject;
        if (emailMessage) {
          emailMessage.value = samples[id].message;
          updateCounters();
        }
        hideError();
        if (resultPanel) resultPanel.style.display = "none";
      });
    }
  });

  // Clear Button
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      if (emailSubject) emailSubject.value = "";
      if (emailMessage) emailMessage.value = "";
      updateCounters();
      hideError();
      if (resultPanel) resultPanel.style.display = "none";
    });
  }

  // Error Alert Helpers
  function showError(msg) {
    if (!errorAlert || !errorMessageText) return;
    errorMessageText.textContent = msg;
    errorAlert.style.display = "flex";
  }

  function hideError() {
    if (!errorAlert) return;
    errorAlert.style.display = "none";
  }

  // Submit Handler for Detection
  if (detectForm) {
    detectForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      hideError();

      const subject = emailSubject ? emailSubject.value.trim() : "";
      const message = emailMessage ? emailMessage.value.trim() : "";

      if (!message) {
        showError("Message content cannot be empty. Please enter or paste text to analyze.");
        if (emailMessage) emailMessage.focus();
        return;
      }

      if (message.length > 50000) {
        showError("Message exceeds the maximum allowable length of 50,000 characters.");
        return;
      }

      // Set Loading State
      if (analyzeBtn) analyzeBtn.disabled = true;
      if (btnIcon) btnIcon.style.display = "none";
      if (btnSpinner) btnSpinner.style.display = "inline-block";
      if (btnLabel) btnLabel.textContent = "Analyzing Message...";

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ subject, message })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
          showError(data.error || "An error occurred while evaluating the message.");
          return;
        }

        // Render Results
        renderPredictionResult(data);

      } catch (err) {
        showError("Network error: Failed to reach the classification server. Please check your connection.");
        console.error("Prediction request failed:", err);
      } finally {
        // Reset Loading State
        if (analyzeBtn) analyzeBtn.disabled = false;
        if (btnIcon) btnIcon.style.display = "inline-block";
        if (btnSpinner) btnSpinner.style.display = "none";
        if (btnLabel) btnLabel.textContent = "Analyze Message";
      }
    });
  }

  function renderPredictionResult(data) {
    if (!resultPanel) return;

    const isSpam = data.prediction === "spam";
    const spamProb = data.probabilities.spam || 0;
    const hamProb = data.probabilities.ham || 0;

    // Apply visual themes
    resultPanel.className = `result-panel ${isSpam ? "spam" : "ham"}`;

    if (verdictIcon) {
      verdictIcon.innerHTML = isSpam
        ? '<i class="bi bi-shield-x"></i>'
        : '<i class="bi bi-shield-check"></i>';
    }

    if (verdictHeading) {
      verdictHeading.textContent = isSpam
        ? "Potential Spam Detected"
        : "Likely Legitimate Message";
    }

    if (verdictExplanation) {
      verdictExplanation.textContent = isSpam
        ? "The machine learning model evaluated this message and classified it as potential spam based on vocabulary patterns."
        : "The machine learning model evaluated this message and classified it as legitimate communication.";
    }

    if (verdictBadge) {
      verdictBadge.textContent = data.label;
    }

    if (confidencePercentage) {
      confidencePercentage.textContent = `${data.confidence}% Confidence`;
    }

    if (probBarHam) probBarHam.style.width = `${hamProb}%`;
    if (probBarSpam) probBarSpam.style.width = `${spamProb}%`;
    if (probHamVal) probHamVal.textContent = `${hamProb}%`;
    if (probSpamVal) probSpamVal.textContent = `${spamProb}%`;

    if (riskAssessmentTag) riskAssessmentTag.textContent = data.risk_level;
    if (analyzedCharCount) analyzedCharCount.textContent = `${data.char_count.toLocaleString()} chars`;
    if (analyzedWordCount) analyzedWordCount.textContent = `${data.word_count.toLocaleString()} words`;

    if (analyzedContentBox) {
      analyzedContentBox.textContent = data.analyzed_text;
      analyzedContentBox.style.display = "none";
      if (accordionIcon) accordionIcon.className = "bi bi-chevron-down";
    }

    resultPanel.style.display = "block";
    resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Accordion Toggle for Analyzed Text
  if (toggleAnalyzedText && analyzedContentBox) {
    toggleAnalyzedText.addEventListener("click", () => {
      const isVisible = analyzedContentBox.style.display === "block";
      analyzedContentBox.style.display = isVisible ? "none" : "block";
      if (accordionIcon) {
        accordionIcon.className = isVisible ? "bi bi-chevron-down" : "bi bi-chevron-up";
      }
    });
  }

  // Analyze Another Button
  if (analyzeAnotherBtn) {
    analyzeAnotherBtn.addEventListener("click", () => {
      if (emailSubject) emailSubject.value = "";
      if (emailMessage) {
        emailMessage.value = "";
        emailMessage.focus();
      }
      updateCounters();
      if (resultPanel) resultPanel.style.display = "none";
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  // -------------------------------------------------------------------------
  // Modal Handling for View Details
  // -------------------------------------------------------------------------
  const detailModal = document.getElementById("detailModal");
  const modalCloseBtn = document.getElementById("modalCloseBtn");
  const modalCloseBtn2 = document.getElementById("modalCloseBtn2");
  const modalContent = document.getElementById("modalContent");

  document.querySelectorAll(".view-detail-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!detailModal || !modalContent) return;

      const subject = btn.dataset.subject || "(No subject provided)";
      const snippet = btn.dataset.snippet || "";
      const label = btn.dataset.label || "N/A";
      const confidence = btn.dataset.confidence || "0";
      const spamProb = btn.dataset.spam || "0";
      const hamProb = btn.dataset.ham || "0";
      const time = btn.dataset.time || "";

      const isSpam = label.toUpperCase().includes("SPAM") && !label.toUpperCase().includes("NOT");

      modalContent.innerHTML = `
        <div style="margin-bottom: 16px;">
          <div style="font-size: 11.5px; text-transform: uppercase; color: var(--text-muted); font-weight: 700;">Analyzed At</div>
          <div style="font-size: 13.5px; color: var(--text-main);">${escapeHtml(time)}</div>
        </div>

        <div style="margin-bottom: 16px;">
          <div style="font-size: 11.5px; text-transform: uppercase; color: var(--text-muted); font-weight: 700;">Subject</div>
          <div style="font-size: 14px; font-weight: 600; color: var(--text-main);">${escapeHtml(subject)}</div>
        </div>

        <div style="margin-bottom: 16px;">
          <div style="font-size: 11.5px; text-transform: uppercase; color: var(--text-muted); font-weight: 700;">Message Snippet</div>
          <div style="background-color: #F8FAFC; border: 1px solid var(--border-color); padding: 12px; border-radius: 8px; font-size: 13px; font-family: monospace; white-space: pre-wrap; word-break: break-word;">${escapeHtml(snippet)}</div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
          <div style="background-color: ${isSpam ? '#FEE2E2' : '#DCFCE7'}; padding: 12px; border-radius: 8px;">
            <div style="font-size: 11.5px; color: ${isSpam ? '#991B1B' : '#166534'}; font-weight: 700;">CLASSIFICATION</div>
            <div style="font-size: 18px; font-weight: 800; color: ${isSpam ? '#DC2626' : '#16A34A'};">${escapeHtml(label)}</div>
          </div>
          <div style="background-color: #EFF6FF; padding: 12px; border-radius: 8px;">
            <div style="font-size: 11.5px; color: #1E40AF; font-weight: 700;">CONFIDENCE SCORE</div>
            <div style="font-size: 18px; font-weight: 800; color: #2563EB;">${escapeHtml(confidence)}%</div>
          </div>
        </div>

        <div style="margin-top: 14px; font-size: 12.5px; display: flex; justify-content: space-between;">
          <span style="color: var(--success-green); font-weight: 600;">Legitimate (Ham): ${escapeHtml(hamProb)}%</span>
          <span style="color: var(--spam-red); font-weight: 600;">Spam: ${escapeHtml(spamProb)}%</span>
        </div>
      `;

      detailModal.classList.add("open");
    });
  });

  function closeDetailModal() {
    if (detailModal) detailModal.classList.remove("open");
  }

  if (modalCloseBtn) modalCloseBtn.addEventListener("click", closeDetailModal);
  if (modalCloseBtn2) modalCloseBtn2.addEventListener("click", closeDetailModal);
  window.addEventListener("click", (e) => {
    if (e.target === detailModal) closeDetailModal();
  });

  function escapeHtml(text) {
    if (!text) return "";
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return String(text).replace(/[&<>"']/g, (m) => map[m]);
  }
});
