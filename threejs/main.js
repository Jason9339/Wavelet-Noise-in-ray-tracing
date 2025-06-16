import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// 場景與相機
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(0, 2, 8);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.0, 0);
controls.update();

// WASD 控制變數
const keys = {
  w: false,
  a: false,
  s: false,
  d: false
};

const moveSpeed = 0.1;

// 噪聲配置
const noiseConfig = {
  wavelet: {
    octaves: [3, 4, 5],
    variants: ['2d', '3d_sliced', '3d_projected']
  },
  perlin: {
    octaves: [3, 4, 5],
    variants: ['2d', '3d_sliced']
  }
};

// 當前選擇
let currentNoise = {
  type: 'wavelet',    // 'wavelet' 或 'perlin'
  octave: 4,          // 3, 4, 5
  variant: '2d'       // '2d', '3d_sliced', '3d_projected'
};

// 紋理緩存
const textureCache = new Map();
let sphere = null;
let floor = null;

// 生成文件名
function getFileName(type, octave, variant) {
  const variantMap = {
    '2d': '2d',
    '3d_sliced': '3d_sliced', 
    '3d_projected': '3d_projected'
  };
  
  if (type === 'wavelet') {
    return `wavelet_noise_${variantMap[variant]}_octave${octave}.json`;
  } else {
    return `perlin_noise_${variantMap[variant]}_octave${octave}.json`;
  }
}

// 載入噪聲紋理的通用函數
async function loadNoiseTexture(filename) {
  // 檢查緩存
  if (textureCache.has(filename)) {
    return textureCache.get(filename);
  }

  try {
    const response = await fetch(`./result_json/${filename}`);
    const noiseData = await response.json();
    
    const { width, height, data, original_range } = noiseData;
    
    console.log(`載入成功: ${filename}`);
    console.log(`- 尺寸: ${width}x${height}`);
    console.log(`- 範圍: [${original_range.min.toFixed(4)}, ${original_range.max.toFixed(4)}]`);
    
    // 將歸一化數據 [0,1] 轉換為 RGBA 格式
    const uint8Array = new Uint8Array(width * height * 4);
    
    for (let i = 0; i < data.length; i++) {
      const value = Math.floor(data[i] * 255); // 0-1 -> 0-255
      
      const rgbaIndex = i * 4;
      uint8Array[rgbaIndex + 0] = value; // R
      uint8Array[rgbaIndex + 1] = value; // G
      uint8Array[rgbaIndex + 2] = value; // B
      uint8Array[rgbaIndex + 3] = 255;   // A
    }
    
    // 創建紋理
    const texture = new THREE.DataTexture(uint8Array, width, height, THREE.RGBAFormat);
    texture.needsUpdate = true;
    texture.magFilter = THREE.LinearFilter;
    texture.minFilter = THREE.LinearFilter;
    
    const result = { texture, info: original_range };
    
    // 添加到緩存
    textureCache.set(filename, result);
    
    return result;
    
  } catch (error) {
    console.error(`載入失敗: ${filename}`, error);
    return null;
  }
}

// 創建材質
function createMaterial(texture, repeatCount = 1, isForSphere = false) {
  const material = new THREE.MeshBasicMaterial({ 
    map: texture,
    side: THREE.DoubleSide
  });
  
  if (texture) {
    if (isForSphere) {
      texture.wrapS = THREE.ClampToEdgeWrapping;
      texture.wrapT = THREE.ClampToEdgeWrapping;
      texture.repeat.set(1, 1);
      texture.offset.set(0, 0);
    } else {
      texture.wrapS = THREE.RepeatWrapping;
      texture.wrapT = THREE.RepeatWrapping;
      texture.repeat.set(repeatCount, repeatCount);
      texture.offset.set(0, 0);
    }
    texture.needsUpdate = true;
  }
  
  return material;
}

// 更新材質
async function updateMaterials() {
  const filename = getFileName(currentNoise.type, currentNoise.octave, currentNoise.variant);
  const result = await loadNoiseTexture(filename);
  
  if (result && sphere && floor) {
    // 更新球體材質
    const sphereTexture = result.texture.clone();
    sphere.material = createMaterial(sphereTexture, 1, true);
    
    // 更新地板材質
    const floorTexture = result.texture.clone();
    floor.material = createMaterial(floorTexture, 9, false);
    
    updateInfo(result.info);
    return true;
  }
  
  return false;
}

// 切換噪聲類型
function switchNoiseType(type) {
  currentNoise.type = type;
  
  // 確保當前 octave 在新類型中可用
  const availableOctaves = noiseConfig[type].octaves;
  if (!availableOctaves.includes(currentNoise.octave)) {
    currentNoise.octave = availableOctaves[0];
  }
  
  // 確保當前變體在新類型中可用
  const availableVariants = noiseConfig[type].variants;
  if (!availableVariants.includes(currentNoise.variant)) {
    currentNoise.variant = availableVariants[0];
  }
  
  updateMaterials();
}

// 切換 octave
function switchOctave(octave) {
  const availableOctaves = noiseConfig[currentNoise.type].octaves;
  if (availableOctaves.includes(octave)) {
    currentNoise.octave = octave;
    updateMaterials();
  }
}

// 切換變體
function switchVariant(variant) {
  const availableVariants = noiseConfig[currentNoise.type].variants;
  if (availableVariants.includes(variant)) {
    currentNoise.variant = variant;
    updateMaterials();
  }
}

// 創建場景物件
async function createScene() {
  // 初始載入預設紋理
  const filename = getFileName(currentNoise.type, currentNoise.octave, currentNoise.variant);
  const result = await loadNoiseTexture(filename);
  
  if (result) {
    // 球體
    const sphereTexture = result.texture.clone();
    sphere = new THREE.Mesh(
      new THREE.SphereGeometry(1, 128, 128), 
      createMaterial(sphereTexture, 1, true)
    );
    sphere.position.y = 1;
    scene.add(sphere);
    
    // 地面
    const floorTexture = result.texture.clone();
    floor = new THREE.Mesh(
      new THREE.PlaneGeometry(40, 40, 256, 256), 
      createMaterial(floorTexture, 9, false)
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0;
    scene.add(floor);
    
    console.log(`使用 ${currentNoise.type} 噪聲 octave ${currentNoise.octave} ${currentNoise.variant}`);
  } else {
    console.log('載入失敗，無法創建材質');
  }
  
  // 添加環境光
  const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
  scene.add(ambientLight);
  
  // 添加方向光
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(5, 5, 5);
  scene.add(directionalLight);
  
  updateInfo(result ? result.info : null);
}

// 初始化場景
createScene();

// 更新信息面板
function updateInfo(noiseInfo = null) {
  const info = document.getElementById('info');
  if (!info) return;
  
  const typeDisplayName = currentNoise.type === 'wavelet' ? 'Wavelet' : 'Perlin';
  const variantDisplayName = {
    '2d': '2D',
    '3d_sliced': '3D Sliced',
    '3d_projected': '3D Projected'
  }[currentNoise.variant];
  
  if (noiseInfo) {
    info.innerHTML = `
      <strong>${typeDisplayName} Noise Visualization</strong><br>
      - Octave: ${currentNoise.octave}<br>
      - 變體: ${variantDisplayName}<br>
      - 尺寸: 256x256<br>
      - 數據範圍: [${noiseInfo.min.toFixed(3)}, ${noiseInfo.max.toFixed(3)}]<br>
      - 平均值: ${noiseInfo.mean.toFixed(3)}, 標準差: ${noiseInfo.std.toFixed(3)}<br>
      - 地板重複: 9x9<br>

      <br><strong>控制:</strong><br>
      - WASD: 移動鏡頭<br>
      - 滑鼠: 旋轉視角，滾輪縮放<br>
      - 數字鍵 1/2: 切換 Wavelet/Perlin<br>
      - Q/E: 切換 Octave (${noiseConfig[currentNoise.type].octaves.join('/')})<br>
      - Z/X/C: 切換變體 (2D/3D切片/3D投影)<br>
      - R: 重置視角
    `;
  } else {
    info.innerHTML = `
      <strong>載入中...</strong><br>
      正在讀取 ${typeDisplayName} 噪聲數據...
    `;
  }
}

// 鍵盤事件監聽
window.addEventListener('keydown', (event) => {
  const key = event.key.toLowerCase();
  
  // WASD 移動控制
  if (keys.hasOwnProperty(key)) {
    keys[key] = true;
  }
  
  // 切換噪聲類型
  if (key === '1') {
    switchNoiseType('wavelet');
  } else if (key === '2') {
    switchNoiseType('perlin');
  }
  
  // 切換 octave
  else if (key === 'q') {
    const octaves = noiseConfig[currentNoise.type].octaves;
    const currentIndex = octaves.indexOf(currentNoise.octave);
    const nextIndex = (currentIndex - 1 + octaves.length) % octaves.length;
    switchOctave(octaves[nextIndex]);
  } else if (key === 'e') {
    const octaves = noiseConfig[currentNoise.type].octaves;
    const currentIndex = octaves.indexOf(currentNoise.octave);
    const nextIndex = (currentIndex + 1) % octaves.length;
    switchOctave(octaves[nextIndex]);
  }
  
  // 切換變體
  else if (key === 'z') {
    switchVariant('2d');
  } else if (key === 'x') {
    switchVariant('3d_sliced');
  } else if (key === 'c') {
    switchVariant('3d_projected');
  }
  
  // 重置視角
  else if (key === 'r') {
    camera.position.set(0, 2, 8);
    controls.target.set(0, 1, 0);
    controls.update();
  }
});

window.addEventListener('keyup', (event) => {
  const key = event.key.toLowerCase();
  if (keys.hasOwnProperty(key)) {
    keys[key] = false;
  }
});

// 鏡頭移動邏輯
function updateCameraMovement() {
  const cameraDirection = new THREE.Vector3();
  camera.getWorldDirection(cameraDirection);
  
  const cameraRight = new THREE.Vector3();
  cameraRight.crossVectors(cameraDirection, camera.up).normalize();
  
  if (keys.w) {
    camera.position.addScaledVector(cameraDirection, moveSpeed);
  }
  if (keys.s) {
    camera.position.addScaledVector(cameraDirection, -moveSpeed);
  }
  if (keys.a) {
    camera.position.addScaledVector(cameraRight, -moveSpeed);
  }
  if (keys.d) {
    camera.position.addScaledVector(cameraRight, moveSpeed);
  }
  
  // 更新軌道控制的目標點
  controls.target.copy(camera.position).addScaledVector(cameraDirection, 5);
  controls.update();
}

// Resize handler
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Animate
function animate() {
  requestAnimationFrame(animate);
  updateCameraMovement();
  renderer.render(scene, camera);
}
animate();

// 添加說明文字
const info = document.createElement('div');
info.id = 'info';
info.style.position = 'absolute';
info.style.top = '10px';
info.style.left = '10px';
info.style.color = 'white';
info.style.fontFamily = 'Arial, sans-serif';
info.style.fontSize = '14px';
info.style.backgroundColor = 'rgba(0,0,0,0.85)';
info.style.padding = '15px';
info.style.borderRadius = '8px';
info.style.border = '1px solid rgba(255,255,255,0.2)';
info.style.maxWidth = '400px';
info.style.lineHeight = '1.4';
info.innerHTML = `
  <strong>載入中...</strong><br>
  正在讀取噪聲數據...
`;
document.body.appendChild(info);
