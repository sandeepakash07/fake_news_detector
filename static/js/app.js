// app.js - Custom JavaScript for Fake News Detector

document.addEventListener('DOMContentLoaded', function() {
    // Character counter for text input
    const contentTextarea = document.getElementById('content');
    if (contentTextarea) {
        const createCounter = () => {
            const counterDiv = document.createElement('div');
            counterDiv.className = 'text-sm text-gray-500 text-right mt-1';
            counterDiv.id = 'char-counter';
            contentTextarea.parentNode.insertBefore(counterDiv, contentTextarea.nextSibling);
            return counterDiv;
        };

        const counterDiv = createCounter();
        
        const updateCounter = () => {
            const count = contentTextarea.value.length;
            counterDiv.textContent = `${count} characters`;
            
            // Simple validation feedback
            if (count < 100) {
                counterDiv.classList.add('text-red-500');
                counterDiv.classList.remove('text-gray-500');
                counterDiv.textContent += ' (Consider adding more content for better analysis)';
            } else {
                counterDiv.classList.remove('text-red-500');
                counterDiv.classList.add('text-gray-500');
            }
        };
        
        contentTextarea.addEventListener('input', updateCounter);
        updateCounter(); // Initial count
    }
    
    // Form validation
    const textForm = document.querySelector('form[action*="analyze"]');
    if (textForm) {
        textForm.addEventListener('submit', function(event) {
            const content = contentTextarea.value.trim();
            if (content.length < 50) {
                event.preventDefault();
                alert('Please enter at least 50 characters for a meaningful analysis.');
            }
        });
    }
    
    // URL form validation
    const urlInput = document.getElementById('url');
    const urlForm = urlInput ? urlInput.closest('form') : null;
    
    if (urlForm) {
        urlForm.addEventListener('submit', function(event) {
            const url = urlInput.value.trim();
            if (!url || !url.match(/^https?:\/\/.+\..+/)) {
                event.preventDefault();
                alert('Please enter a valid URL starting with http:// or https://');
            }
        });
    }
    
    // Auto-dismiss messages after 5 seconds
    const messages = document.querySelectorAll('.messages .message');
    messages.forEach(message => {
        setTimeout(() => {
            message.style.opacity = '0';
            setTimeout(() => {
                message.style.display = 'none';
            }, 500);
        }, 5000);
    });
});