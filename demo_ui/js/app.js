document.addEventListener('DOMContentLoaded', () => {
    const uploadTrigger = document.getElementById('upload-trigger');
    const loadingOverlay = document.getElementById('loading-overlay');

    if (uploadTrigger && loadingOverlay) {
        uploadTrigger.addEventListener('click', () => {
             // Show Loading Overlay
            loadingOverlay.classList.add('active');
            
            // Simulate processing time
            setTimeout(() => {
                // Navigate to report page (determine if we are in EN or AR)
                const isEnglish = document.documentElement.lang === 'en';
                if (isEnglish) {
                    window.location.href = 'report.html';
                } else {
                    window.location.href = 'report.html';
                }
            }, 2500); // 2.5 seconds mock delay
        });
    }
});
