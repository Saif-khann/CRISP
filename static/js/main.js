/**
 * CRISP - Construction Recognition & Intelligence for Stage Progress
 * Modern Client-side Controller with Smooth Async Validation & Comparison
 */

// The CSRF token is rendered into a meta tag by base.html. Requests that
// build FormData by hand (rather than from a <form> containing the hidden
// field) must send it as a header instead.
const CSRF_TOKEN = document.querySelector('meta[name="csrf-token"]')?.content || '';

document.addEventListener('DOMContentLoaded', function() {
    // --------------------------------------------------------------------------
    // Image Validation Form Submission
    // --------------------------------------------------------------------------
    const validationForm = document.getElementById('validationForm');
    const resultArea = document.getElementById('resultArea');
    const submitBtn = document.getElementById('submitBtn');

    if (validationForm) {
        validationForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(validationForm);

            // UI Loading state
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Running Ensemble AI Inference...';
            }

            resultArea.innerHTML = `
                <div class="text-center py-5">
                    <div class="spinner-grow text-primary mb-3" style="width: 3rem; height: 3rem;" role="status"></div>
                    <h6 class="fw-bold mb-1">Evaluating CNN Ensemble</h6>
                    <p class="small text-muted mb-0">Computing voting across MobileNetV2 (30%), InceptionV3 (40%), and VGG16 (30%)...</p>
                </div>
            `;

            try {
                const response = await fetch(validationForm.action, {
                    method: 'POST',
                    headers: { 'X-CSRFToken': CSRF_TOKEN },
                    body: formData
                });
                const result = await response.json();

                if (result.success) {
                    const stageConf = parseFloat(result.confidence_scores?.['Stage Confidence'] || 95);
                    const globalConf = parseFloat(result.confidence_scores?.['Global Stage Confidence'] || 96);

                    let html = `
                        <div class="animate-fade-in">
                            <div class="d-flex align-items-center justify-content-between mb-3 pb-2 border-bottom">
                                <span class="badge badge-success px-3 py-1 fs-6">
                                    <i class="fas fa-check-circle me-1"></i> Stage Verified
                                </span>
                                <small class="text-muted"><i class="fas fa-clock me-1"></i>${result.timestamp || 'Just now'}</small>
                            </div>

                            <div class="mb-3">
                                <label class="small text-muted text-uppercase fw-bold">Classified Stage & Sub-Stage</label>
                                <h5 class="fw-bold text-primary mb-1">${result.primary_stage?.toUpperCase()}</h5>
                                <p class="fw-semibold text-secondary mb-0">${(result.specific_classification || '').replace(/_/g, ' ')}</p>
                            </div>

                            <!-- Confidence Scores -->
                            <div class="mb-3 p-3 rounded bg-light border border-light">
                                <div class="mb-2">
                                    <div class="d-flex justify-content-between small fw-semibold mb-1">
                                        <span>Sub-stage confidence</span>
                                        <span class="text-primary">${stageConf.toFixed(1)}%</span>
                                    </div>
                                    <div class="progress" style="height: 8px;">
                                        <div class="progress-bar bg-primary" style="width: ${stageConf}%;"></div>
                                    </div>
                                </div>
                                <div>
                                    <div class="d-flex justify-content-between small fw-semibold mb-1">
                                        <span>Overall confidence</span>
                                        <span class="text-success">${globalConf.toFixed(1)}%</span>
                                    </div>
                                    <div class="progress" style="height: 8px;">
                                        <div class="progress-bar bg-success" style="width: ${globalConf}%;"></div>
                                    </div>
                                </div>
                                <p class="small text-muted mt-2 mb-0" style="opacity: 0.8;">How sure the model is about the sub-stage, and about the broader construction stage.</p>
                            </div>

                            <!-- AI Narrative -->
                            ${result.description ? `
                            <div class="mb-3 p-3 rounded" style="background: var(--info-subtle); border-left: 4px solid var(--info);">
                                <h6 class="fw-bold text-info mb-1" style="font-size: 0.85rem;">
                                    <i class="fas fa-note-sticky me-1"></i> AI notes
                                </h6>
                                <p class="small text-secondary mb-0" style="line-height: 1.5;">${result.description}</p>
                            </div>
                            ` : ''}

                            <div class="d-flex gap-2 mt-3">
                                <button type="button" class="btn btn-sm btn-outline-primary flex-fill" onclick="location.reload()">
                                    <i class="fas fa-rotate me-1"></i> Refresh Dashboard
                                </button>
                                <a href="/generate_report/${result.validation_id}?project_id=${document.getElementById('project-id')?.value}" class="btn btn-sm btn-dark flex-fill">
                                    <i class="fas fa-file-pdf me-1"></i> Download PDF
                                </a>
                            </div>
                        </div>
                    `;
                    resultArea.innerHTML = html;
                } else if (result.mismatch) {
                    resultArea.innerHTML = `
                        <div class="alert alert-warning animate-fade-in">
                            <h6 class="fw-bold"><i class="fas fa-triangle-exclamation me-1"></i> Stage mismatch</h6>
                            <p class="small mb-3">${result.message}</p>
                            <p class="small text-muted mb-0">Pick the matching stage above, or tick “Record this validation even if the AI flags a stage mismatch” to save it anyway.</p>
                        </div>
                    `;
                } else {
                    resultArea.innerHTML = `
                        <div class="alert alert-danger animate-fade-in">
                            <h6 class="fw-bold"><i class="fas fa-circle-exclamation me-1"></i> Validation Error</h6>
                            <p class="small mb-0">${result.error || 'Validation failed. Please try again.'}</p>
                        </div>
                    `;
                }
            } catch (err) {
                resultArea.innerHTML = `
                    <div class="alert alert-danger animate-fade-in">
                        <h6 class="fw-bold">Connection Error</h6>
                        <p class="small mb-0">${err.message}</p>
                    </div>
                `;
            } finally {
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-wand-magic-sparkles me-1"></i> Analyze & Validate with AI';
                }
            }
        });
    }

    // --------------------------------------------------------------------------
    // Historical Milestone Comparison
    // --------------------------------------------------------------------------
    const compareBtn = document.getElementById('compareBtn');
    if (compareBtn) {
        compareBtn.addEventListener('click', async function() {
            const projectId = document.getElementById('project-id')?.value;
            const prevId = document.getElementById('previous-validation')?.value;
            const currId = document.getElementById('current-validation')?.value;
            const compResults = document.getElementById('comparison-results');

            if (!prevId || !currId) {
                alert('Please select both a baseline and current milestone to compare.');
                return;
            }

            compareBtn.disabled = true;
            compareBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';

            const formData = new FormData();
            formData.append('project_id', projectId);
            formData.append('previous_doc_id', prevId);
            formData.append('current_doc_id', currId);

            try {
                const response = await fetch('/compare', {
                    method: 'POST',
                    headers: { 'X-CSRFToken': CSRF_TOKEN },
                    body: formData
                });
                const data = await response.json();

                if (data.success) {
                    const statusBadge = data.progress_status === 'advanced' ? 'badge-success' : data.progress_status === 'same' ? 'badge-warning' : 'badge-danger';
                    
                    compResults.innerHTML = `
                        <div class="card p-3 bg-light border border-light animate-fade-in">
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <h6 class="fw-bold text-uppercase small text-muted mb-0">Milestone Comparison</h6>
                                <span class="badge ${statusBadge} text-uppercase">${data.progress_status}</span>
                            </div>

                            <div class="row g-3 mb-3">
                                <div class="col-md-6">
                                    <div class="p-3 bg-white rounded border">
                                        <small class="text-muted d-block fw-bold mb-1">BASELINE MILESTONE</small>
                                        <h6 class="fw-bold text-primary mb-1">${data.previous.stage?.toUpperCase()} (${(data.previous.sub_stage || '').replace(/_/g, ' ')})</h6>
                                        <p class="small text-muted mb-0">Overall: ${data.previous.overall_progress?.toFixed(1)}% complete</p>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="p-3 bg-white rounded border">
                                        <small class="text-muted d-block fw-bold mb-1">TARGET MILESTONE</small>
                                        <h6 class="fw-bold text-success mb-1">${data.current.stage?.toUpperCase()} (${(data.current.sub_stage || '').replace(/_/g, ' ')})</h6>
                                        <p class="small text-muted mb-0">Overall: ${data.current.overall_progress?.toFixed(1)}% complete</p>
                                    </div>
                                </div>
                            </div>

                            <div class="p-3 bg-white rounded border">
                                <h6 class="small fw-bold text-secondary mb-2"><i class="fas fa-chart-line me-1"></i> Summary</h6>
                                <p class="small text-muted mb-0" style="white-space: pre-line;">${data.progress_message}</p>
                            </div>
                        </div>
                    `;
                    compResults.style.display = 'block';
                } else {
                    alert(data.error || 'Failed to compare progress');
                }
            } catch (err) {
                alert('Comparison error: ' + err.message);
            } finally {
                compareBtn.disabled = false;
                compareBtn.innerHTML = '<i class="fas fa-code-compare me-1"></i> Compare';
            }
        });
    }
});
