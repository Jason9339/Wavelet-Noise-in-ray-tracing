// WebGL Shader 工具類
class ShaderUtils {
    static createShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const error = gl.getShaderInfoLog(shader);
            gl.deleteShader(shader);
            throw new Error(`著色器編譯錯誤: ${error}`);
        }
        
        return shader;
    }
    
    static createProgram(gl, vertexShaderSource, fragmentShaderSource) {
        const vertexShader = this.createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
        const fragmentShader = this.createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
        
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const error = gl.getProgramInfoLog(program);
            gl.deleteProgram(program);
            throw new Error(`程式連結錯誤: ${error}`);
        }
        
        return program;
    }
    
    static getAttribLocation(gl, program, name) {
        const location = gl.getAttribLocation(program, name);
        if (location === -1) {
            console.warn(`找不到屬性: ${name}`);
        }
        return location;
    }
    
    static getUniformLocation(gl, program, name) {
        const location = gl.getUniformLocation(program, name);
        if (location === null) {
            console.warn(`找不到uniform: ${name}`);
        }
        return location;
    }
    
    static createBuffer(gl, data) {
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
        return buffer;
    }
    
    static setupVertexAttribute(gl, location, size, type = gl.FLOAT, normalized = false, stride = 0, offset = 0) {
        gl.enableVertexAttribArray(location);
        gl.vertexAttribPointer(location, size, type, normalized, stride, offset);
    }
}

// Vertex Shader 原始碼
const vertexShaderSource = `
    attribute vec2 a_position;
    varying vec2 v_uv;
    
    void main() {
        v_uv = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
`;

// Fragment Shader 原始碼 - 簡化的雜訊風格渲染
const fragmentShaderSource = `
    precision highp float;
    
    varying vec2 v_uv;
    
    uniform float u_time;
    uniform vec2 u_resolution;
    uniform vec3 u_cameraPos;
    uniform float u_sphereY;
    uniform float u_noiseFreq;
    
    // 常數定義
    const float PI = 3.14159265359;
    const float MAX_DIST = 100.0;
    const int MAX_STEPS = 128;
    const float EPSILON = 0.001;
    
    // 隨機數產生
    float random(vec2 st) {
        return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
    }
    
    // 改良的 Perlin 雜訊實現
    vec2 fade(vec2 t) {
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    }
    
    float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        
        float a = random(i);
        float b = random(i + vec2(1.0, 0.0));
        float c = random(i + vec2(0.0, 1.0));
        float d = random(i + vec2(1.0, 1.0));
        
        vec2 u = fade(f);
        
        return mix(a, b, u.x) + 
               (c - a) * u.y * (1.0 - u.x) + 
               (d - b) * u.x * u.y;
    }
    
    // 改良的 3D 雜訊實現
    float noise3D(vec3 p) {
        // 使用更好的3D雜訊組合方式
        vec3 i = floor(p);
        vec3 f = fract(p);
        
        // 8個頂點的雜訊值
        float n000 = random(i.xy + i.z * 57.0);
        float n001 = random(i.xy + (i.z + 1.0) * 57.0);
        float n010 = random((i.xy + vec2(0.0, 1.0)) + i.z * 57.0);
        float n011 = random((i.xy + vec2(0.0, 1.0)) + (i.z + 1.0) * 57.0);
        float n100 = random((i.xy + vec2(1.0, 0.0)) + i.z * 57.0);
        float n101 = random((i.xy + vec2(1.0, 0.0)) + (i.z + 1.0) * 57.0);
        float n110 = random((i.xy + vec2(1.0, 1.0)) + i.z * 57.0);
        float n111 = random((i.xy + vec2(1.0, 1.0)) + (i.z + 1.0) * 57.0);
        
        // 3D插值
        vec3 u = f * f * (3.0 - 2.0 * f);
        
        float nx00 = mix(n000, n100, u.x);
        float nx01 = mix(n001, n101, u.x);
        float nx10 = mix(n010, n110, u.x);
        float nx11 = mix(n011, n111, u.x);
        
        float nxy0 = mix(nx00, nx10, u.y);
        float nxy1 = mix(nx01, nx11, u.y);
        
        return mix(nxy0, nxy1, u.z);
    }
    
    // 多層次 3D 雜訊
    float fbm3D(vec3 p) {
        float value = 0.0;
        float amplitude = 0.5;
        float frequency = 1.0;
        
        // 增加到6個octave，與地板一致
        for (int i = 0; i < 6; i++) {
            value += amplitude * noise3D(p * frequency);
            amplitude *= 0.5;
            frequency *= 2.0;
        }
        
        return value;
    }
    
    // 多層次 Perlin 雜訊 (地板用)
    float fbm2D(vec2 p) {
        float value = 0.0;
        float amplitude = 0.5;
        float frequency = 1.0;
        
        for (int i = 0; i < 6; i++) {
            value += amplitude * noise(p * frequency);
            amplitude *= 0.5;
            frequency *= 2.0;
        }
        
        return value;
    }
    
    // 多層次 Perlin 雜訊 (保持向後相容)
    float perlinNoise(vec2 p) {
        return fbm2D(p * u_noiseFreq);
    }
    
    // SDF 函數
    float sphereSDF(vec3 p, vec3 center, float radius) {
        return length(p - center) - radius;
    }
    
    float planeSDF(vec3 p, vec4 plane) {
        return dot(p, plane.xyz) + plane.w;
    }
    
    // 場景 SDF
    vec2 sceneSDF(vec3 p) {
        // 單一球體
        float sphere = sphereSDF(p, vec3(0.0, u_sphereY, 0.0), 0.8);
        
        // 地板
        float plane = planeSDF(p, vec4(0.0, 1.0, 0.0, 0.0));
        
        // 回傳距離和材質ID (球體=1.0, 地板=2.0)
        if (sphere < plane) {
            return vec2(sphere, 1.0);
        } else {
            return vec2(plane, 2.0);
        }
    }
    
    // Ray marching
    vec2 rayMarch(vec3 ro, vec3 rd) {
        float t = 0.0;
        float matId = 0.0;
        
        for (int i = 0; i < MAX_STEPS; i++) {
            vec3 p = ro + t * rd;
            vec2 result = sceneSDF(p);
            float dist = result.x;
            matId = result.y;
            
            if (dist < EPSILON) {
                break;
            }
            
            t += dist;
            
            if (t > MAX_DIST) {
                matId = 0.0;
                break;
            }
        }
        
        return vec2(t, matId);
    }
    
    // 計算法向量
    vec3 getNormal(vec3 p) {
        vec2 e = vec2(EPSILON, 0.0);
        return normalize(vec3(
            sceneSDF(p + e.xyy).x - sceneSDF(p - e.xyy).x,
            sceneSDF(p + e.yxy).x - sceneSDF(p - e.yxy).x,
            sceneSDF(p + e.yyx).x - sceneSDF(p - e.yyx).x
        ));
    }
    
    // 取得雜訊強度
    float getNoiseIntensity(vec3 pos, float matId) {
        if (matId == 1.0) {
            // 球體 - 使用3D FBM雜訊，頻率與地板相近
            float noise = fbm3D(pos * u_noiseFreq * 0.8);
            return 0.3 + 0.7 * abs(noise); // 與地板相似的變化範圍
        } else if (matId == 2.0) {
            // 地板 - 使用2D FBM雜訊
            float noise = fbm2D(pos.xz * u_noiseFreq * 0.8);
            return 0.3 + 0.7 * abs(noise); // 與球體相同的變化範圍
        }
        return 0.5;
    }
    
    void main() {
        vec2 uv = (v_uv * 2.0 - 1.0) * vec2(u_resolution.x / u_resolution.y, 1.0);
        
        // 相機設定
        vec3 ro = u_cameraPos;
        vec3 target = vec3(0.0, 0.0, 0.0);
        vec3 up = vec3(0.0, 1.0, 0.0);
        
        vec3 w = normalize(ro - target);
        vec3 u = normalize(cross(up, w));
        vec3 v = cross(w, u);
        
        vec3 rd = normalize(uv.x * u + uv.y * v - 1.8 * w);
        
        // Ray marching
        vec2 result = rayMarch(ro, rd);
        float t = result.x;
        float matId = result.y;
        
        // 背景顏色（天空）
        vec3 color = vec3(0.9, 0.95, 1.0);
        
        if (matId > 0.0 && t < MAX_DIST) {
            vec3 pos = ro + t * rd;
            vec3 normal = getNormal(pos);
            
            // 取得雜訊強度
            float noiseValue = getNoiseIntensity(pos, matId);
            
            // 基礎陰影（簡單的法向量dot產品）
            float lightDir = dot(normal, normalize(vec3(0.5, 1.0, 0.3)));
            float baseBrightness = 0.4 + 0.6 * max(0.0, lightDir);
            
            // 使用雜訊調製顏色
            float intensity = baseBrightness * noiseValue;
            
            // 創造黑白漸層效果
            color = vec3(intensity);
            
            // 確保球體不會太暗
            if (matId == 1.0) {
                // 球體增加一些基礎亮度
                color = max(color, vec3(0.2));
            }
        }
        
        // 簡單的對比度增強
        color = smoothstep(0.0, 1.0, color);
        
        // 輸出最終顏色
        gl_FragColor = vec4(color, 1.0);
    }
`; 