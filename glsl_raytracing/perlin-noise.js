// Perlin 雜訊 JavaScript 實現
class PerlinNoise {
    constructor(seed = 0) {
        this.seed = seed;
        this.permutation = this.generatePermutation();
    }
    
    generatePermutation() {
        const p = [];
        for (let i = 0; i < 256; i++) {
            p[i] = i;
        }
        
        // 使用種子進行洗牌
        let random = this.seededRandom(this.seed);
        for (let i = 255; i > 0; i--) {
            const j = Math.floor(random() * (i + 1));
            [p[i], p[j]] = [p[j], p[i]];
        }
        
        // 複製陣列以避免索引越界
        for (let i = 0; i < 256; i++) {
            p[256 + i] = p[i];
        }
        
        return p;
    }
    
    seededRandom(seed) {
        let state = seed;
        return function() {
            state = (state * 9301 + 49297) % 233280;
            return state / 233280;
        };
    }
    
    fade(t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }
    
    lerp(a, b, t) {
        return a + t * (b - a);
    }
    
    grad(hash, x, y) {
        const h = hash & 15;
        const u = h < 8 ? x : y;
        const v = h < 4 ? y : h === 12 || h === 14 ? x : 0;
        return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
    }
    
    noise(x, y) {
        const X = Math.floor(x) & 255;
        const Y = Math.floor(y) & 255;
        
        x -= Math.floor(x);
        y -= Math.floor(y);
        
        const u = this.fade(x);
        const v = this.fade(y);
        
        const A = this.permutation[X] + Y;
        const AA = this.permutation[A];
        const AB = this.permutation[A + 1];
        const B = this.permutation[X + 1] + Y;
        const BA = this.permutation[B];
        const BB = this.permutation[B + 1];
        
        return this.lerp(
            this.lerp(
                this.grad(this.permutation[AA], x, y),
                this.grad(this.permutation[BA], x - 1, y),
                u
            ),
            this.lerp(
                this.grad(this.permutation[AB], x, y - 1),
                this.grad(this.permutation[BB], x - 1, y - 1),
                u
            ),
            v
        );
    }
    
    // 分層雜訊 (Fractal Brownian Motion)
    fbm(x, y, octaves = 4, persistence = 0.5) {
        let value = 0;
        let amplitude = 1;
        let frequency = 1;
        let maxValue = 0;
        
        for (let i = 0; i < octaves; i++) {
            value += this.noise(x * frequency, y * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= 2;
        }
        
        return value / maxValue;
    }
}

// 向量和矩陣工具
class Vec3 {
    constructor(x = 0, y = 0, z = 0) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
    
    static add(a, b) {
        return new Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
    
    static subtract(a, b) {
        return new Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
    }
    
    static multiply(v, s) {
        return new Vec3(v.x * s, v.y * s, v.z * s);
    }
    
    static dot(a, b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    
    static cross(a, b) {
        return new Vec3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }
    
    length() {
        return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
    }
    
    normalize() {
        const len = this.length();
        if (len > 0) {
            return new Vec3(this.x / len, this.y / len, this.z / len);
        }
        return new Vec3(0, 0, 0);
    }
    
    toArray() {
        return [this.x, this.y, this.z];
    }
}

// 相機控制器
class CameraController {
    constructor() {
        this.position = new Vec3(0, 2, 3);
        this.target = new Vec3(0, 0, 0);
        this.up = new Vec3(0, 1, 0);
        this.fov = 60;
        this.aspect = 1;
        this.near = 0.1;
        this.far = 100;
    }
    
    setPosition(x, y, z) {
        this.position = new Vec3(x, y, z);
    }
    
    setTarget(x, y, z) {
        this.target = new Vec3(x, y, z);
    }
    
    getViewMatrix() {
        const forward = Vec3.subtract(this.target, this.position).normalize();
        const right = Vec3.cross(forward, this.up).normalize();
        const actualUp = Vec3.cross(right, forward);
        
        return {
            position: this.position,
            forward: forward,
            right: right,
            up: actualUp
        };
    }
}

// 動畫工具
class AnimationUtils {
    static easeInOut(t) {
        return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    }
    
    static smoothStep(edge0, edge1, x) {
        const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
        return t * t * (3 - 2 * t);
    }
    
    static lerp(a, b, t) {
        return a + (b - a) * t;
    }
    
    static clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }
}

// 效能監控
class PerformanceMonitor {
    constructor() {
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 0;
        this.frameTime = 0;
    }
    
    update() {
        const currentTime = performance.now();
        this.frameTime = currentTime - this.lastTime;
        this.lastTime = currentTime;
        
        this.frameCount++;
        if (this.frameCount % 60 === 0) {
            this.fps = Math.round(1000 / this.frameTime);
        }
    }
    
    getFPS() {
        return this.fps;
    }
    
    getFrameTime() {
        return this.frameTime.toFixed(2);
    }
} 