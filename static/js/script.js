// Document ready function
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Form input validation
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.checkValidity()) {
                this.classList.remove('is-invalid');
            }
        });
    });
    
    // Storage validation
    const hddInput = document.getElementById('HDD');
    const ssdInput = document.getElementById('SSD');
    
    if (hddInput && ssdInput) {
        [hddInput, ssdInput].forEach(input => {
            input.addEventListener('change', function() {
                const hdd = parseInt(hddInput.value) || 0;
                const ssd = parseInt(ssdInput.value) || 0;
                
                if (hdd <= 0 && ssd <= 0) {
                    hddInput.classList.add('is-invalid');
                    ssdInput.classList.add('is-invalid');
                } else {
                    hddInput.classList.remove('is-invalid');
                    ssdInput.classList.remove('is-invalid');
                }
            });
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
});