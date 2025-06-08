import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as dat from 'https://cdn.jsdelivr.net/npm/dat.gui/+esm';

// 場景與相機
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(0, 1.5, 5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.0, 0);
controls.update();

// ✅ Uniforms（不再需要 camDist）
const uniforms = {
  octaveCount: { value: 4 },
  frequency: { value: 20.0 }
};

// ✅ ShaderMaterial（無 blur）
const material = new THREE.ShaderMaterial({
  uniforms,
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform int octaveCount;
    uniform float frequency;
    varying vec2 vUv;

    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
    }

    float noise(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      vec2 u = f * f * (3.0 - 2.0 * f);
      return mix(
        mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
        u.y
      );
    }

    float fbm(vec2 p, int octaves) {
      float value = 0.0;
      float amplitude = 0.5;
      float freq = 1.0;
      for (int i = 0; i < 8; i++) {
        if (i >= octaves) break;
        value += amplitude * noise(p * freq);
        freq *= 2.0;
        amplitude *= 0.5;
      }
      return value;
    }

    void main() {
      float n = fbm(vUv * frequency, octaveCount);
      gl_FragColor = vec4(vec3(n), 1.0);  // ✅ 無 blur，直接輸出 noise
    }
  `
});

// 幾何物件
const sphere = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 64), material);
sphere.position.y = 1;
scene.add(sphere);

const floor = new THREE.Mesh(new THREE.PlaneGeometry(100, 100), material);
floor.rotation.x = -Math.PI / 2;
floor.position.y = 0;
scene.add(floor);

// Resize handler
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Animate
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ✅ GUI
const gui = new dat.GUI();
gui.add(uniforms.octaveCount, 'value', 1, 20, 1).name('Octaves');
gui.add(uniforms.frequency, 'value', 1, 100, 1).name('Frequency');
