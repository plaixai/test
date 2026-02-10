const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let overlayWindow;
let avatarGalleryWindow;
let pythonProcess;
let gradioURL = null;
let appPath = null; // Global app path for file checking
let resourcesPath = null; // Global resources path for file access

function createWindow() {
  // Determine paths based on whether app is packaged or not
  const isPackaged = app.isPackaged;
  appPath = isPackaged ? path.dirname(app.getPath('exe')) : path.join(__dirname, '..'); // Set global

  // For installed apps, resources are alongside the executable
  // For development, use the app directory
  if (isPackaged) {
    // Check if we're running from an installed location (not win-unpacked)
    const exeDir = path.dirname(app.getPath('exe'));
    const isInstalled = exeDir.includes('Programs') || exeDir.includes('PLAIX AI') || exeDir.includes('Program Files');

    if (isInstalled) {
      // Installed version: resources are in the same directory as executable
      resourcesPath = exeDir;
      console.log('Detected installed app, using exe directory for resources');
    } else {
      // Portable/win-unpacked version: use process.resourcesPath
      resourcesPath = process.resourcesPath;
      console.log('Detected portable app, using process.resourcesPath');
    }
  } else {
    // In development, use the parent directory (go up one level from electron)
    resourcesPath = path.resolve(__dirname, '..');
    console.log('Development mode, using parent directory for resources:', resourcesPath);
  }

  console.log('Path resolution:');
  console.log('  Is Packaged:', isPackaged);
  console.log('  App Path:', appPath);
  console.log('  Resources Path:', resourcesPath);
  console.log('  Executable Dir:', path.dirname(app.getPath('exe')));
  
  // List files in resources path for debugging
  try {
    const fs = require('fs');
    if (fs.existsSync(resourcesPath)) {
      const files = fs.readdirSync(resourcesPath);
      console.log('  Files in resources path:', files);
    } else {
      console.log('  Resources path does not exist!');
    }
  } catch (e) {
    console.log('  Error listing files:', e.message);
  }
  
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false, // Disable CORS/CORB for localhost
      allowRunningInsecureContent: true,
      experimentalFeatures: true
    },
    icon: path.join(__dirname, '../icon.png'),
    title: 'PLAIX AI - AI Commentary System'
  });

  // Configure session to allow localhost
  mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': ['default-src * \'unsafe-inline\' \'unsafe-eval\' data: blob:;']
      }
    });
  });

  // DevTools disabled for production
  // mainWindow.webContents.openDevTools();

  // Load landing page immediately
  const loadingPage = path.join(__dirname, 'loading.html');
  console.log('Loading splash screen from:', loadingPage);
  mainWindow.loadFile(loadingPage);

  // Start Python backend
  startPythonBackend();

  // Wait for overlay API to be ready on port 7862, then create overlay
  let overlayCreated = false;
  let checkOverlayAPI = setInterval(() => {
    const http = require('http');
    const options = {
      hostname: '127.0.0.1',
      port: 7862,
      path: '/api/overlay_status',
      method: 'GET',
      timeout: 1000
    };

    const req = http.request(options, (res) => {
      if (res.statusCode === 200) {
        clearInterval(checkOverlayAPI);
        clearTimeout(timeoutId);
        overlayCreated = true;
        console.log(`✓ Overlay API ready on port 7862`);
        console.log('✓ Creating overlay window...');
        
        // Show a simple info page in main window
        const infoHTML = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>PLAIX AI - Overlay Mode</title>
  <style>
    body { 
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      color: #fff; 
      font-family: 'Segoe UI', sans-serif; 
      padding: 60px; 
      text-align: center;
      margin: 0;
    }
    .container { max-width: 800px; margin: 0 auto; }
    h1 { color: #667eea; font-size: 3em; margin-bottom: 20px; }
    p { font-size: 1.3em; line-height: 1.8; margin: 20px 0; }
    .highlight { color: #4ecdc4; font-weight: bold; }
    .status { 
      background: rgba(102, 126, 234, 0.1); 
      padding: 30px; 
      border-radius: 15px; 
      margin: 30px 0; 
      border: 2px solid rgba(102, 126, 234, 0.3);
    }
    .emoji { font-size: 3em; }
  </style>
</head>
<body>
  <div class="container">
    <div class="emoji">🎭</div>
    <h1>PLAIX AI - Overlay Mode</h1>
    
    <div class="status">
      <p>✅ <span class="highlight">Overlay API Running</span></p>
      <p>✅ <span class="highlight">Commentary System Active</span></p>
      <p>✅ <span class="highlight">Avatar Overlay Ready</span></p>
    </div>
    
    <p>All settings and controls are now in the <span class="highlight">overlay window</span>.</p>
    <p>Look for the overlay on your screen - you can drag it to any monitor!</p>
    <p>Press the <span class="highlight">settings button</span> in the overlay to customize.</p>
    
    <p style="margin-top: 40px; font-size: 0.9em; opacity: 0.7;">
      This window can be minimized - all functionality is in the overlay.
    </p>
  </div>
</body>
</html>
        `;
        mainWindow.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(infoHTML));
        
        // Overlay will be created by overlay.flag file watcher
      }
    });

    req.on('error', () => {
      // API not ready yet, will retry
    });

    req.on('timeout', () => {
      req.destroy();
    });

    req.end();
  }, 500);

  // Timeout fallback after 120 seconds (AI models can take time to load)
  let timeoutId = setTimeout(() => {
    clearInterval(checkOverlayAPI);
    
    // Don't show error if overlay was already created successfully
    if (overlayCreated) {
      console.log('✓ Overlay already running, ignoring timeout');
      return;
    }
    
    console.error('⚠️ Overlay API not ready after 180s');
    const errorHTML = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Backend Error</title>
  <style>
    body { background:#1a1a2e; color:#fff; font-family:sans-serif; padding:40px; }
    .container { max-width: 800px; margin: 0 auto; }
    h1 { color: #ff6b6b; }
    .error-details { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; margin: 20px 0; }
    .troubleshooting { background: rgba(255,193,7,0.1); padding: 20px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 20px 0; }
    button { margin-top:20px; padding:12px 24px; cursor:pointer; background:#667eea; border:none; color:white; border-radius:8px; font-size:16px; transition: background 0.3s; }
    button:hover { background:#5568d3; }
    .secondary { background:#4ecdc4; }
    .secondary:hover { background:#45b8b0; }
    code { background: #2d2d44; padding: 5px 10px; border-radius: 5px; font-family: monospace; }
  </style>
</head>
<body>
  <div class="container">
    <h1>❌ Python Backend Failed to Start</h1>
    
    <div class="error-details">
      <p><strong>Status:</strong> Could not connect to Overlay API on port 7862</p>
      <p><strong>App Path:</strong> ${appPath || 'unknown'}</p>
    </div>
    
    <div class="troubleshooting">
      <h3>⚠️ First Time Setup Required</h3>
      <p><strong>The packaged app needs Python dependencies installed!</strong></p>
      <p>Open PowerShell or Command Prompt and run:</p>
      <code style="display:block; margin:10px 0; padding:15px;">
cd "${appPath || 'C:\\Users\\YourName\\AppData\\Local\\Programs\\PLAIX AI'}"<br>
pip install -r requirements.txt
      </code>
      <p>Then restart the application.</p>
    </div>
    
    <p>The Python backend (plaix.py) started but couldn't launch the API server. This could be due to:</p>
    
    <h3>Common Issues:</h3>
    <ul style="text-align:left;">
      <li><strong>Python not installed</strong> - Download from <a href="https://www.python.org/downloads/" style="color:#667eea;">python.org</a></li>
      <li><strong>Python not in PATH</strong> - Reinstall Python and check "Add Python to PATH"</li>
      <li><strong>Missing dependencies</strong> - Run <code>pip install -r requirements.txt</code> in the app folder</li>
      <li><strong>Missing models</strong> - Ensure YOLO weights and Kokoro TTS files are present</li>
      <li><strong>Port conflict</strong> - Another app might be using port 7862</li>
      <li><strong>Script crash</strong> - Check for import errors or missing files</li>
    </ul>
    
    <h3>Manual Testing:</h3>
    <p>1. Open PowerShell or Command Prompt<br>
    2. Navigate to: <code>${appPath || 'C:\\Users\\YourName\\AppData\\Local\\Programs\\PLAIX AI'}</code><br>
    3. Run: <code>python plaix.py</code><br>
    4. If it starts successfully, return to this app and click Retry</p>
    
    <div style="margin-top: 30px;">
      <button onclick="location.reload()">🔄 Retry Connection</button>
      <button class="secondary" onclick="openAppFolder()">📁 Open App Folder</button>
    </div>
  </div>
  
  <script>
    function openAppFolder() {
      if (typeof require !== 'undefined') {
        const { shell } = require('electron');
        shell.openPath('${appPath || 'C:\\\\Users\\\\YourName\\\\AppData\\\\Local\\\\Programs\\\\PLAIX AI'}');
      }
    }
  </script>
</body>
</html>`;
    mainWindow.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(errorHTML));
  }, 180000); // 180 seconds timeout to allow AI models to load

  // Monitor for overlay.flag file to manually trigger overlay creation
  const fs = require('fs');
  const overlayFlagCheck = setInterval(() => {
    const flagPath = path.join(resourcesPath, 'overlay.flag');
    if (fs.existsSync(flagPath)) {
      console.log('📝 Overlay flag detected, creating overlay...');
      fs.unlinkSync(flagPath); // Delete the flag file
      
      // Clear the API check and timeout since overlay is working
      clearInterval(checkOverlayAPI);
      clearTimeout(timeoutId);
      overlayCreated = true;
      
      // Show success screen in main window
      const infoHTML = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>PLAIX AI - Overlay Mode</title>
  <style>
    body { 
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      color: #fff; 
      font-family: 'Segoe UI', sans-serif; 
      padding: 60px; 
      text-align: center;
      margin: 0;
    }
    .container { max-width: 800px; margin: 0 auto; }
    h1 { color: #667eea; font-size: 3em; margin-bottom: 20px; }
    p { font-size: 1.3em; line-height: 1.8; margin: 20px 0; }
    .highlight { color: #4ecdc4; font-weight: bold; }
    .status { 
      background: rgba(102, 126, 234, 0.1); 
      padding: 30px; 
      border-radius: 15px; 
      margin: 30px 0; 
      border: 2px solid rgba(102, 126, 234, 0.3);
    }
    .emoji { font-size: 3em; }
  </style>
</head>
<body>
  <div class="container">
    <div class="emoji">🎭</div>
    <h1>PLAIX AI - Overlay Mode</h1>
    
    <div class="status">
      <p>✅ <span class="highlight">Overlay API Running</span></p>
      <p>✅ <span class="highlight">Commentary System Active</span></p>
      <p>✅ <span class="highlight">Avatar Overlay Ready</span></p>
    </div>
    
    <p>All settings and controls are now in the <span class="highlight">overlay window</span>.</p>
    <p>Look for the overlay on your screen - you can drag it to any monitor!</p>
    <p>Press the <span class="highlight">settings button</span> in the overlay to customize.</p>
    
    <p style="margin-top: 40px; font-size: 0.9em; opacity: 0.7;">
      This window can be minimized - all functionality is in the overlay.
    </p>
  </div>
</body>
</html>
      `;
      mainWindow.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(infoHTML));
      
      if (!overlayWindow) {
        createOverlay();
      } else {
        console.log('Overlay already exists');
      }
    }
  }, 1000); // Check every second

  mainWindow.on('closed', () => {
    mainWindow = null;
    clearInterval(overlayFlagCheck);
  });
}

function createOverlay() {
  console.log('Starting createOverlay function');
  const { screen } = require('electron');
  console.log('Required electron screen module');
  const displays = screen.getAllDisplays();
  console.log('Got displays:', displays.length);
  
  // Find the 1920x1080 display or use primary
  let targetDisplay = displays.find(d => d.bounds.width === 1920 && d.bounds.height === 1080);
  
  if (!targetDisplay) {
    // Fallback to primary if no 1920x1080 display found
    targetDisplay = screen.getPrimaryDisplay();
    console.log('⚠️ No 1920x1080 display found, using primary display');
  } else {
    console.log('✓ Using 1920x1080 display for overlay');
  }
  
  // Check if there are multiple displays
  if (displays.length > 1) {
    console.log(`Found ${displays.length} displays.`);
    console.log('Available displays:', displays.map((d, i) => 
      `Display ${i}: ${d.bounds.width}x${d.bounds.height} at (${d.bounds.x}, ${d.bounds.y})`
    ));
  }
  
  const { x, y, width, height } = targetDisplay.bounds;
  
  console.log(`Creating overlay: ${width}x${height} at position (${x}, ${y})`);

  try {
    overlayWindow = new BrowserWindow({
      x: x,
      y: y,
      width: width,
      height: height,
      transparent: true,
      frame: false,
      alwaysOnTop: true,
      skipTaskbar: true,  // Hide from taskbar to not interfere
      resizable: false,
      movable: true,  // Allow moving to other monitors
      focusable: false,  // Don't steal focus from games
      show: true,  // Show immediately for debugging
      webPreferences: {
        nodeIntegration: true,  // Enable for IPC
        contextIsolation: false,  // Disable for simpler IPC
        webSecurity: false
      }
    });
    console.log('BrowserWindow created successfully');
  } catch (error) {
    console.error('Failed to create BrowserWindow:', error);
    return;
  }

  // Set to always be on top but don't interfere with fullscreen apps
  overlayWindow.setAlwaysOnTop(true, 'pop-up-menu');
  
  // On Windows, allow overlay to show over fullscreen without blocking the game
  if (process.platform === 'win32') {
    overlayWindow.setVisibleOnAllWorkspaces(false);
    // Remove the hook that was causing issues
    // overlayWindow.hookWindowMessage(0x0084, () => {
    //   overlayWindow.setEnabled(false);
    //   setImmediate(() => overlayWindow.setEnabled(true));
    //   return true;
    // });
  }

  // Allow clicking buttons but click-through elsewhere
  overlayWindow.setIgnoreMouseEvents(false);  // Changed to false

  // Load overlay HTML (handle both dev and packaged app paths)
  const overlayPath = path.join(__dirname, 'overlay.html');
  console.log('Loading overlay from:', overlayPath);
  overlayWindow.loadFile(overlayPath);

  // Log overlay console messages to main process
  overlayWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
    console.log(`[OVERLAY] ${message}`);
  });

  overlayWindow.on('closed', () => {
    overlayWindow = null;
  });

  // Show overlay after content is loaded, without stealing focus
  overlayWindow.once('ready-to-show', () => {
    // overlayWindow.showInactive();  // Show without stealing focus - disabled for debugging
    console.log('✓ Overlay ready to show (but not showing for debugging)');
  });

  console.log('✓ Overlay window created');
  console.log('💡 Tip: Right-click the overlay window to move to another monitor');
}

// IPC handlers for overlay controls
ipcMain.on('toggle-click-through', (event, enabled) => {
  if (overlayWindow) {
    overlayWindow.setIgnoreMouseEvents(enabled, { forward: true });
    console.log(`Click-through ${enabled ? 'enabled' : 'disabled'}`);
  }
});

ipcMain.on('settings-panel-opened', (event) => {
  if (overlayWindow) {
    overlayWindow.setFocusable(true);
    overlayWindow.focus();
    console.log('Settings panel opened - focusing overlay');
  }
});

ipcMain.on('settings-panel-closed', (event) => {
  if (overlayWindow) {
    // Just blur but keep focusable so buttons remain clickable
    overlayWindow.blur();
    console.log('Settings panel closed - blurred (still clickable)');
  }
});

ipcMain.on('chat-box-focused', (event) => {
  if (overlayWindow) {
    overlayWindow.setFocusable(true);
    overlayWindow.focus();
    console.log('Chat box focused - focusing overlay for input');
  }
});

ipcMain.on('chat-box-blurred', (event) => {
  if (overlayWindow) {
    // Just blur but keep focusable so buttons remain clickable
    overlayWindow.blur();
    console.log('Chat box blurred - blurred (still clickable)');
  }
});

// Open avatar gallery window
ipcMain.on('open-avatar-gallery', (event) => {
  console.log('Opening avatar gallery...');
  
  if (avatarGalleryWindow) {
    avatarGalleryWindow.focus();
    return;
  }

  avatarGalleryWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false
    },
    title: 'Choose Your Personality - PLAIX',
    backgroundColor: '#0a0a1a',
    frame: true,
    resizable: true
  });

  avatarGalleryWindow.loadFile(path.join(__dirname, 'avatar_gallery.html'));
  
  avatarGalleryWindow.on('closed', () => {
    avatarGalleryWindow = null;
    console.log('Avatar gallery closed');
  });
});

// Close avatar gallery
ipcMain.on('close-avatar-gallery', (event) => {
  if (avatarGalleryWindow) {
    avatarGalleryWindow.close();
  }
});

ipcMain.on('toggle-commentary', (event, enabled) => {
  console.log(`Commentary ${enabled ? 'start' : 'stop'} requested from overlay`);
  if (mainWindow) {
    // Execute JavaScript in main window to click the commentary button
    const buttonId = enabled ? 'component-2' : 'component-3'; // Start/Stop button IDs
    mainWindow.webContents.executeJavaScript(`
      const btn = document.getElementById('${buttonId}');
      if (btn) btn.click();
    `);
  }
});

ipcMain.on('toggle-voice-chat', (event, enabled) => {
  console.log(`Voice chat ${enabled ? 'start' : 'stop'} requested from overlay`);
  if (mainWindow) {
    // Execute JavaScript in main window to click the voice chat button
    const buttonId = enabled ? 'component-4' : 'component-5'; // Voice Start/Stop button IDs  
    mainWindow.webContents.executeJavaScript(`
      const btn = document.getElementById('${buttonId}');
      if (btn) btn.click();
    `);
  }
});

// Provide app paths to renderer process for animation library
ipcMain.handle('get-app-paths', async () => {
  return {
    appPath: appPath,
    resourcesPath: resourcesPath,
    isPackaged: app.isPackaged
  };
});

// Open URL in a new Electron window (for content that can't be iframed)
let contentWindow = null;
ipcMain.on('open-content-window', (event, url, title) => {
  console.log(`Loading content in main window: ${title} - ${url}`);
  
  // Use the main window to display content
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.loadURL(url);
    mainWindow.setTitle(title || 'PLAIX Content');
    mainWindow.show();
    mainWindow.focus();
  }
});

function startPythonBackend() {
  // Use global paths set in createWindow
  const exePath = path.join(resourcesPath, 'PLAIX-AI.exe');
  const scriptPath = path.join(resourcesPath, 'plaix.py');
  let workingDir = resourcesPath; // Use resources path as working directory (let for reassignment)

  console.log('=== PLAIX AI Starting ===');
  console.log('Is Packaged:', app.isPackaged);
  console.log('App Path:', appPath);
  console.log('Resources Path:', resourcesPath);
  console.log('EXE path:', exePath);
  console.log('Script path:', scriptPath);
  console.log('Working directory:', workingDir);
  
  const fs = require('fs');
  
  // Check for required files - prefer executable in packaged app
  // Also check in subdirectories in case files are nested
  const requiredFiles = [
    { name: 'plaix.py', path: scriptPath },
    { name: 'echo_ai.py', path: path.join(resourcesPath, 'echo_ai.py') },
    { name: 'requirements.txt', path: path.join(resourcesPath, 'requirements.txt') },
    { name: 'kokoro-v1.0.onnx', path: path.join(resourcesPath, 'kokoro-v1.0.onnx') },
    { name: 'yolov11x-rocketleague-best.pt', path: path.join(resourcesPath, 'yolov11x-rocketleague-best.pt') }
  ];
  
  // Also check in subdirectories (resources, app, etc.)
  const possiblePaths = [
    resourcesPath,
    path.join(resourcesPath, 'resources'),
    path.join(resourcesPath, 'app'),
    appPath,
    path.dirname(app.getPath('exe')), // Executable directory
    path.dirname(path.dirname(app.getPath('exe'))) // Parent of executable directory
  ];
  
  console.log('Checking for files in possible locations:');
  possiblePaths.forEach(p => {
    try {
      if (fs.existsSync(p)) {
        const files = fs.readdirSync(p);
        console.log(`  ${p}: ${files.length} files`);
      } else {
        console.log(`  ${p}: directory not found`);
      }
    } catch (e) {
      console.log(`  ${p}: error accessing - ${e.message}`);
    }
  });
  
  const missingFiles = requiredFiles.filter(file => !fs.existsSync(file.path));
  
  // If files are missing, try to find them in the directory tree
  if (missingFiles.length > 0) {
    console.log('Some files missing, searching directory tree...');
    const foundFiles = {};
    
    function searchDirectory(dirPath, maxDepth = 3) {
      if (maxDepth <= 0) return;
      
      try {
        const items = fs.readdirSync(dirPath);
        for (const item of items) {
          const itemPath = path.join(dirPath, item);
          const stat = fs.statSync(itemPath);
          
          if (stat.isFile()) {
            // Check if this file is one we're looking for
            for (const reqFile of requiredFiles) {
              if (item === reqFile.name && !foundFiles[reqFile.name]) {
                foundFiles[reqFile.name] = itemPath;
                console.log(`  Found ${reqFile.name} at: ${itemPath}`);
              }
            }
          } else if (stat.isDirectory() && !item.startsWith('.') && item !== 'node_modules') {
            searchDirectory(itemPath, maxDepth - 1);
          }
        }
      } catch (e) {
        // Ignore permission errors
      }
    }
    
    // Search from the executable directory
    const exeDir = path.dirname(app.getPath('exe'));
    searchDirectory(exeDir);
    
    // If we found some files, update the paths
    for (const reqFile of requiredFiles) {
      if (foundFiles[reqFile.name]) {
        reqFile.path = foundFiles[reqFile.name];
      }
    }
    
    // Re-check for missing files
    const stillMissing = requiredFiles.filter(file => !fs.existsSync(file.path));
    if (stillMissing.length < missingFiles.length) {
      console.log(`Found ${missingFiles.length - stillMissing.length} additional files via search`);
    }
  }
  
  if (missingFiles.length > 0) {
    console.error('[ERROR] Missing required files:', missingFiles.map(f => f.name));
    console.error('App may not function correctly');
    console.log('Files checked:');
    requiredFiles.forEach(file => {
      console.log(`  ${file.name}: ${file.path} -> ${fs.existsSync(file.path) ? 'EXISTS' : 'MISSING'}`);
    });
  } else {
    console.log('[OK] All required files present');
  }
  
  // Use Python script only (no executable for packaged apps)
  const exeExists = fs.existsSync(exePath);
  const scriptExists = fs.existsSync(scriptPath);
  
  let useExecutable = false;
  let finalPath = scriptPath;
  
  // Prefer .exe in production, .py in development
  if (app.isPackaged && exeExists) {
    useExecutable = true;
    finalPath = exePath;
    console.log('[STARTUP] Using compiled executable (standalone mode)');
  } else if (scriptExists) {
    useExecutable = false;
    finalPath = scriptPath;
    // In dev mode, ensure working directory is the parent directory (where all files are)
    workingDir = resourcesPath;
    console.log('[STARTUP] Using Python script (requires Python installed)');
  } else {
    console.error('ERROR: Neither PLAIX-AI.exe nor plaix.py found');
    console.error('EXE path:', exePath);
    console.error('Script path:', scriptPath);
    
    // Try development fallback - look for files in the original development directory
    const devScriptPath = path.join(__dirname, '..', 'plaix.py');
    if (fs.existsSync(devScriptPath)) {
      console.log('Found development version, using fallback path:', devScriptPath);
      finalPath = devScriptPath;
      workingDir = path.dirname(devScriptPath);
    } else {
      if (mainWindow) {
        const errorPath = path.join(__dirname, 'error.html');
        mainWindow.loadFile(errorPath, { query: { path: appPath || 'unknown', error: 'missing_script' } });
      }
      return;
    }
  }
  
  console.log('✓ All required files present');
  console.log(`Using Python script: ${finalPath}`);
  
  // Delete old gradio_port.txt to prevent stale port detection
  const os = require('os');
  const portFilePaths = [
    path.join(resourcesPath, 'gradio_port.txt'),
    path.join(os.tmpdir(), 'plaix_rl_gradio_port.txt'),
    path.join(appPath, 'gradio_port.txt')
  ];

  portFilePaths.forEach(portFilePath => {
    if (fs.existsSync(portFilePath)) {
      try {
        fs.unlinkSync(portFilePath);
        console.log(`🗑️ Deleted old port file: ${portFilePath}`);
      } catch (e) {
        console.log(`⚠️ Could not delete port file: ${portFilePath}`);
      }
    }
  });
  
  // Clean up any existing processes first
  if (process.platform === 'win32') {
    spawn('taskkill', ['/IM', 'PLAIX-AI.exe', '/FI', 'WINDOWTITLE eq PLAIX*', '/F'], {
      shell: true
    });
    spawn('taskkill', ['/IM', 'python.exe', '/FI', 'WINDOWTITLE eq plaix*', '/F'], {
      shell: true
    });
  }
  
  // Wait a moment for cleanup
  setTimeout(() => {
    if (useExecutable) {
      // Launch the compiled executable directly
      console.log('Launching compiled executable...');
      console.log(`Executable path: ${finalPath}`);
      console.log(`Working dir: ${workingDir}`);
      
      pythonProcess = spawn(finalPath, [], {
        cwd: workingDir,
        stdio: 'pipe',
        shell: false,
        windowsHide: false,  // Show console window for debugging
        env: { 
          ...process.env,
          PLAIX_RESOURCES: resourcesPath
        }
      });
      
      console.log('Executable process spawned, PID:', pythonProcess.pid);
      setupPythonHandlers();
      
      // Check if process is still alive after 5 seconds
      setTimeout(() => {
        if (pythonProcess && !pythonProcess.killed) {
          try {
            process.kill(pythonProcess.pid, 0);
            console.log('✓ Executable process is still running after 5 seconds');
          } catch (e) {
            console.error('❌ Executable process died within 5 seconds');
            if (mainWindow) {
              const errorPath = path.join(__dirname, 'error.html');
              mainWindow.loadFile(errorPath, { query: { path: appPath || 'unknown', error: 'exe_crash' } });
            }
          }
        }
      }, 5000);
    } else {
      // Test if Python is available first
      console.log('Testing Python availability...');
      const testPython = spawn('python', ['-c', 'import sys; print(sys.executable)'], { stdio: 'pipe' });
    
    let pythonPath = '';
    let pythonError = '';
    
    testPython.stdout.on('data', (data) => {
      pythonPath += data.toString().trim();
    });
    
    testPython.stderr.on('data', (data) => {
      pythonError += data.toString();
    });
    
    testPython.on('error', (err) => {
      console.error('❌ Python spawn error:', err.message);
      if (mainWindow) {
        const errorPath = path.join(__dirname, 'error.html');
        mainWindow.loadFile(errorPath, { query: { path: appPath || 'unknown', error: 'python_not_found' } });
      }
      return;
    });
    
    testPython.on('exit', (code) => {
      console.log(`Python test exit code: ${code}`);
      console.log(`Python executable path: ${pythonPath}`);
      if (pythonError) {
        console.log(`Python stderr: ${pythonError.trim()}`);
      }
      
      if (code === 0 && pythonPath) {
        console.log('✓ Python is available, starting plaix.py...');
        console.log(`Using Python: ${pythonPath}`);
        console.log(`Script path: ${scriptPath}`);
        console.log(`Working dir: ${workingDir}`);
        
        // Use the full path to Python executable
        pythonProcess = spawn(pythonPath, [scriptPath], {
          cwd: workingDir,
          stdio: 'pipe',
          shell: true,
          windowsHide: false,  // Show console window for debugging
          env: { 
            ...process.env, 
            PYTHONPATH: workingDir,
            PATH: `${path.dirname(pythonPath)};${process.env.PATH}` // Ensure Python DLLs are found
          }
        });
        
        console.log('Python process spawned, PID:', pythonProcess.pid);
        setupPythonHandlers();
        
        // Check if Python process is still alive after 5 seconds
        setTimeout(() => {
          if (pythonProcess && !pythonProcess.killed) {
            try {
              process.kill(pythonProcess.pid, 0); // Signal 0 just checks if process exists
              console.log('✓ Python process is still running after 5 seconds');
            } catch (e) {
              console.error('❌ Python process died within 5 seconds');
              console.error('This indicates the script crashed on startup');
              console.error('Check the Python output above for error messages');
              if (!gradioURL && mainWindow) {
                const errorPath = path.join(__dirname, 'error.html');
                mainWindow.loadFile(errorPath, { query: { path: appPath || 'unknown', error: 'python_crash' } });
              }
            }
          }
        }, 5000);
      } else {
        console.error('❌ Python test failed with code:', code);
        console.error('Python path result:', pythonPath);
        console.error('Python error output:', pythonError);
        if (mainWindow) {
          const errorPath = path.join(__dirname, 'error.html');
          mainWindow.loadFile(errorPath, { query: { path: appPath || 'unknown', error: 'python_test_failed' } });
        }
      }
    });
    } // End of else block for Python script mode
  }, 500); // Wait 500ms for cleanup
}

function setupPythonHandlers() {
  if (!pythonProcess) {
    console.error('❌ setupPythonHandlers called but pythonProcess is null');
    return;
  }
  
  console.log('Setting up Python output handlers...');

  pythonProcess.stdout.on('data', (data) => {
    const output = data.toString();
    console.log(`[PYTHON STDOUT] ${output}`);
    
    // Parse Gradio URL from output - try multiple patterns
    const urlMatch = output.match(/Running on local URL:\s+(http:\/\/[^\s]+)/) ||
                     output.match(/Running on:\s+(http:\/\/[^\s]+)/) ||
                     output.match(/(http:\/\/127\.0\.0\.1:\d+)/) ||
                     output.match(/(http:\/\/localhost:\d+)/);
    
    if (urlMatch && !gradioURL) {
      gradioURL = urlMatch[1];
      console.log(`✓✓✓ Detected Gradio URL: ${gradioURL}`);
    }
  });

  pythonProcess.stderr.on('data', (data) => {
    const output = data.toString();
    
    // Only show as error if it's actually an error (not just warnings)
    if (output.toLowerCase().includes('error') || output.toLowerCase().includes('exception') || output.toLowerCase().includes('traceback')) {
      console.error(`[PYTHON STDERR ERROR] ${output}`);
    } else {
      console.log(`[PYTHON STDERR] ${output}`);
    }
    
    // Also check stderr for Gradio URL (Gradio outputs to stderr sometimes)
    const urlMatch = output.match(/Running on local URL:\s+(http:\/\/[^\s]+)/) ||
                     output.match(/Running on:\s+(http:\/\/[^\s]+)/) ||
                     output.match(/(http:\/\/127\.0\.0\.1:\d+)/) ||
                     output.match(/(http:\/\/localhost:\d+)/);
    
    if (urlMatch && !gradioURL) {
      gradioURL = urlMatch[1];
      console.log(`✓✓✓ Detected Gradio URL from stderr: ${gradioURL}`);
    }
  });

    pythonProcess.on('error', (err) => {
      console.error('❌❌❌ Failed to start Python process:', err);
      console.error('Make sure Python is installed and in PATH');
      console.error('Try running: python --version');
      
      // Show error immediately in UI
      if (mainWindow) {
        const errorPath = path.join(__dirname, 'error.html');
        mainWindow.loadFile(errorPath, { query: { path: appPath || 'unknown' } });
      }
    });

    pythonProcess.on('exit', (code, signal) => {
      console.log(`⚠️ Python process exited with code ${code}, signal ${signal}`);
      if (code !== 0) {
        console.error(`❌ Python script failed with exit code ${code}`);
        console.error('This usually means an import error or missing dependency');
console.error('Try running the script manually: python plaix.py');
        
        // Show error in UI if Gradio URL was never detected
        if (!gradioURL && mainWindow) {
          setTimeout(() => {
            if (!gradioURL) {
              const errorPath = path.join(__dirname, 'error.html');
              mainWindow.loadFile(errorPath, { query: { path: appPath || 'unknown', error: 'python_exit' } });
            }
          }, 1000);
        }
      }
    });
}

// Prevent multiple instances
const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
  console.log('Another instance is already running. Exiting.');
  app.quit();
} else {
  app.on('second-instance', (event, commandLine, workingDirectory) => {
    // Someone tried to run a second instance, focus our window
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.on('ready', async () => {
    // Kill any existing Electron processes before starting
    await cleanupOldProcesses();
    createWindow();
    // Overlay will be created AFTER Gradio server is confirmed running
    // See createWindow() function for the delayed overlay creation
  });
}

async function cleanupOldProcesses() {
  console.log('🧹 Checking for old PLAIX processes...');
  
  if (process.platform === 'win32') {
    try {
      const { execSync } = require('child_process');
      const currentPID = process.pid;
      
      // Get all Electron processes
      try {
        const output = execSync('wmic process where "name=\'electron.exe\'" get ProcessId,CommandLine /format:list', {
          encoding: 'utf8',
          windowsHide: true
        }).toString();
        
        const lines = output.split('\n');
        let electronPIDs = [];
        let commandLine = '';
        
        for (let line of lines) {
          line = line.trim();
          if (line.startsWith('CommandLine=')) {
            commandLine = line.substring(12);
          } else if (line.startsWith('ProcessId=')) {
            const pid = parseInt(line.substring(10));
            if (!isNaN(pid) && pid !== currentPID && commandLine.toLowerCase().includes('plaix')) {
              electronPIDs.push(pid);
            }
          }
        }
        
        // Kill old PLAIX Electron processes
        for (let pid of electronPIDs) {
          console.log(`🔪 Killing old PLAIX Electron process: ${pid}`);
          try {
            execSync(`taskkill /PID ${pid} /F /T`, { windowsHide: true });
          } catch (e) {
            console.log(`  ⚠️ Process ${pid} already terminated`);
          }
        }
        
        if (electronPIDs.length > 0) {
          console.log(`✓ Cleaned up ${electronPIDs.length} old process(es)`);
          // Wait a moment for processes to fully terminate
          await new Promise(resolve => setTimeout(resolve, 1000));
        } else {
          console.log('✓ No old processes found');
        }
      } catch (e) {
        console.log('⚠️ Could not check for old processes (this is ok)');
      }
      
    } catch (err) {
      console.error('Error during cleanup:', err.message);
    }
  }
}

function killPythonProcesses() {
  console.log('Killing Python process tree...');
  
  if (process.platform === 'win32') {
    try {
      // Kill the specific Python process tree if we have the PID
      if (pythonProcess && pythonProcess.pid) {
        spawn('taskkill', ['/pid', pythonProcess.pid.toString(), '/T', '/F'], {
          shell: true
        });
      }
      
      // Also kill any stray plaix.py processes
      spawn('taskkill', ['/IM', 'python.exe', '/FI', 'WINDOWTITLE eq plaix*', '/F'], {
        shell: true
      });
      
      console.log('✓ Python processes terminated');
    } catch (err) {
      console.error('Failed to kill process tree:', err);
      if (pythonProcess) {
        pythonProcess.kill('SIGKILL');
      }
    }
  } else {
    // Unix/Mac - send SIGTERM then SIGKILL
    if (pythonProcess) {
      pythonProcess.kill('SIGTERM');
      setTimeout(() => {
        if (pythonProcess && !pythonProcess.killed) {
          pythonProcess.kill('SIGKILL');
        }
      }, 1000);
    }
  }
  
  pythonProcess = null;
}

app.on('window-all-closed', () => {
  killPythonProcesses();
  app.quit();
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

app.on('will-quit', () => {
  killPythonProcesses();
});

app.on('before-quit', () => {
  killPythonProcesses();
});
