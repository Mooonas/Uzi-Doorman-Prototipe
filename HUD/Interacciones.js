// script.js
const visor = document.querySelector('.visor');

// Cambia la clase CSS del visor para aplicar la transformación geométrica
function setEmotion(emotion) {
    // Limpiamos las clases previas
    visor.className = 'visor'; 
    
    // Si no es neutral, añadimos la clase de la emoción
    if (emotion !== 'neutral') {
        visor.classList.add(emotion);
    }
}

// Sistema de parpadeo orgánico autónomo
function autoBlink() {
    // Guardamos la emoción actual para no sobreescribirla
    const currentEmotion = visor.className;
    
    // Solo parpadea si no está parpadeando ya por comando
    if (!visor.classList.contains('blink')) {
        visor.classList.add('blink');
        
        // El parpadeo dura 150ms
        setTimeout(() => {
            visor.className = currentEmotion; 
        }, 150);
    }

    // Programa el siguiente parpadeo entre 2 y 6 segundos aleatoriamente
    const nextBlink = Math.random() * 4000 + 2000;
    setTimeout(autoBlink, nextBlink);
}

// Iniciar el ciclo orgánico
setTimeout(autoBlink, 3000);