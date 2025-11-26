document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const showRegisterLink = document.getElementById('showRegister');
    const registerCard = document.getElementById('registerCard');

    // Show registration form (hide login) — open register only, do not toggle both
    showRegisterLink.addEventListener('click', function(e) {
        e.preventDefault();
        if (registerCard) {
            registerCard.style.display = 'block';
        }
        const loginCard = document.getElementById('loginCard');
        if (loginCard) {
            loginCard.style.display = 'none';
        }
        // Focus first field in registration form for convenience
        const regUser = document.getElementById('regEmail');
        if (regUser) regUser.focus();
    });

    // Handle login
    loginForm.addEventListener('submit', async function(e) {
        e.preventDefault();
    const username = document.getElementById('email').value;
        const password = document.getElementById('password').value;

        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username,
                    password: password
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Store the token in localStorage
                localStorage.setItem('token', data.token);
                // Redirect to main page
                window.location.href = '/';
            } else {
                alert(data.detail || 'Login failed');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during login');
        }
    });

    // Handle registration
    registerForm.addEventListener('submit', async function(e) {
        e.preventDefault();
    const username = document.getElementById('regEmail').value;
        const password = document.getElementById('regPassword').value;
        const confirmPassword = document.getElementById('regConfirmPassword').value;

        if (password !== confirmPassword) {
            alert('Passwords do not match!');
            return;
        }

        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username,
                    password: password
                })
            });

            const data = await response.json();

            if (response.ok) {
                alert('Registration successful! Please login.');
                registerCard.style.display = 'none';
                // Reveal login card (in case server hid it until registration)
                const loginCard = document.getElementById('loginCard');
                if (loginCard) {
                    loginCard.style.display = 'block';
                }
                // Pre-fill email into login field for convenience
                const emailField = document.getElementById('email');
                if (emailField) emailField.value = username;
            } else {
                alert(data.detail || 'Registration failed');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during registration');
        }
    });

    // Password reveal helper: press/hold to reveal, click to toggle
    function attachPasswordToggle(inputId, btnId) {
        const input = document.getElementById(inputId);
        const btn = document.getElementById(btnId);
        if (!input || !btn) return;
        let locked = false; // click toggles persistent show
        // press and hold
        btn.addEventListener('mousedown', function(e){ e.preventDefault(); input.type = 'text'; });
        btn.addEventListener('mouseup', function(){ if(!locked) input.type = 'password'; });
        btn.addEventListener('mouseleave', function(){ if(!locked) input.type = 'password'; });
        btn.addEventListener('touchstart', function(e){ e.preventDefault(); input.type = 'text'; }, {passive:false});
        btn.addEventListener('touchend', function(){ if(!locked) input.type = 'password'; });
        // click toggles persistent visibility
        btn.addEventListener('click', function(e){ e.preventDefault(); locked = !locked; input.type = locked ? 'text' : 'password'; btn.classList.toggle('active', locked); });
    }

    attachPasswordToggle('password','togglePassword');
    attachPasswordToggle('regPassword','toggleRegPassword');
    attachPasswordToggle('regConfirmPassword','toggleRegConfirmPassword');

});