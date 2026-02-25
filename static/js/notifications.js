  const emIn = document.getElementById('emAddress');
  if (emIn) emIn.onchange = saveNotifPrefs;
  
  // Forzar carga inicial
  setTimeout(loadNotifPrefs, 500);
}

function toggleChart(){
  const b=document.getElementById('chartBody'),btn=document.getElementById('btnToggleChart');
  if(b.style.display==='none'){
    b.style.display='block';
    btn.innerText='▲ Cerrar gráfico';
    if(typeof chartLoaded !== 'undefined' && !chartLoaded) loadChart();
  }else{
    b.style.display='none';
    btn.innerText='▼ Abrir gráfico';
  }
}

function toggleFavs(){
  const b=document.getElementById('favBody'),c=document.getElementById('favChevron');
  if(b.classList.contains('closed')){
    b.classList.remove('closed');b.classList.add('open');
    if(c) c.style.transform='rotate(0deg)';
  }else{
    b.classList.remove('open');b.classList.add('closed');
    if(c) c.style.transform='rotate(-90deg)';
  }
}

let isVip = false;

async function loadNotifPrefs(){
  try{
    const userId = getNotifUserId();
    const r=await fetch(`/api/notifications/prefs?user_id=${userId}`);
    const p=await r.json();
    isVip = !!p.is_vip;
    console.log("VIP Status loaded:", isVip, "for user:", userId);
    
    // UI VIP Status
    const btn = document.getElementById('notifBtn');
    if(btn) {
        if(isVip) {
            btn.innerHTML = '💎 Acceso VIP';
            btn.style.borderColor = '#c9a84c';
            btn.style.color = '#c9a84c';
        } else {
            btn.innerHTML = '🔔 Alertas (Free)';
            btn.style.borderColor = '';
            btn.style.color = '';
        }
    }

    const tgEn = document.getElementById('tgEnabled');
    if(tgEn) tgEn.checked=!!p.telegram_enabled;
    const tgId = document.getElementById('tgChatId');
    if(tgId) tgId.value=p.telegram_chat_id||'';
    const tgF = document.getElementById('tgFields');
    if(tgF) tgF.style.display=p.telegram_enabled?'block':'none';
    
    const emEn = document.getElementById('emEnabled');
    if(emEn) emEn.checked=!!p.email_enabled;
    const emAd = document.getElementById('emAddress');
    if(emAd) emAd.value=p.email_address||'';
    const emF = document.getElementById('emFields');
    if(emF) emF.style.display=p.email_enabled?'block':'none';

    // Bloqueo visual si no es VIP
    const statusBox = document.getElementById('notifStatus');
    if(!isVip) {
        document.querySelectorAll('.notif-channel').forEach(el => {
            el.style.opacity = '0.5';
            el.style.pointerEvents = 'none';
        });
        if(statusBox) {
            statusBox.innerHTML = `
                <div style="display:flex;flex-direction:column;gap:10px;padding:15px;background:rgba(201,168,76,0.05);border-radius:8px;border:1px solid rgba(201,168,76,0.2)">
                    <div style="color:#e8c96d;text-align:center;font-weight:700">👑 Pásate a PRO para activar alertas</div>
                    <div style="display:flex;gap:8px">
                        <input type="text" id="vipCodeInput" placeholder="Introduce código VIP" style="flex:1;padding:8px;border-radius:6px;background:#111;border:1px solid #333;color:#fff;font-family:var(--font-mono);font-size:12px">
                        <button onclick="redeemVipCode()" style="padding:8px 12px;background:#c9a84c;border:none;border-radius:6px;color:#000;font-weight:700;cursor:pointer;font-size:12px">Canjear</button>
                    </div>
                </div>
            `;
        }
    } else {
        document.querySelectorAll('.notif-channel').forEach(el => {
            el.style.opacity = '1';
            el.style.pointerEvents = 'auto';
        });
        if(statusBox) statusBox.innerHTML = '<div style="color:#2dd47e;text-align:center;padding:10px;font-size:12px">💎 Beneficios VIP Activos</div>';
        if(typeof loadNotifStatus !== 'undefined') loadNotifStatus(); 
    }
    if(typeof updateWatchlistUI !== 'undefined') updateWatchlistUI();
  }catch(e){ console.error("Error loading prefs:", e); }
}

async function redeemVipCode() {
    const codeInput = document.getElementById('vipCodeInput');
    const code = codeInput ? codeInput.value.trim() : '';
    if(!code) return;
    
    showToast("Verificando código...", "info");
    try {
        const userId = getNotifUserId();
        const r = await fetch(`/api/notifications/redeem?user_id=${userId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code })
        });
        const d = await r.json();
        if(d.ok) {
            showToast(d.msg, "bullish");
            setTimeout(() => location.reload(), 1000); 
        } else {
            showToast(d.error || "Código inválido", "bearish");
        }
    } catch(e) {
        showToast("Error al conectar con el servidor", "bearish");
    }
}
