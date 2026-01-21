# Qwen3-VL TPU推理服务

基于算能SE7盒子的Qwen3-VL视觉语言模型FastAPI推理服务，支持多并发、本地媒体文件、URL媒体资源和视频处理、API Key认证保护。

## 📋 项目信息

- **模型**: Qwen3-VL-Instruct
- **硬件**: 算能BM1684X TPU (SE7盒子)
- **框架**: FastAPI + Sophon BMRuntime + 多线程并发
- **默认端口**: 8899
- **默认API Key**: `abc@123`
- **版本**: 0.1.0

# 🚀 快速开始
需使用算能 SE7 盒子（BM1684X 芯片），先通过 SD 卡或 OTA 方式安装算能 SDK 环境（[SDK 下载官网](https://developer.sophgo.com/site/index/material/all/all.html)）；Python 环境需≥3.10，推荐通过 Miniconda 创建虚拟环境管理依赖，确保适配 ARM64 架构的边缘计算资源配置。

### 1. 环境准备
#### 1.1 下载预编译模型文件（推荐，无需自行编译）
直接下载算能官方预编译的BM1684X模型文件，省去编译步骤：
```bash
# 准备目录
cd qwen3-vl-sophon-tpu-serving
mkdir -p ./models/qwen3vl_4b

# 安装依赖
pip3 install -r requirements.txt

# 下载1684x 4B模型（最大1K输入, 768x768像素, 视频最长12s/1帧/秒）
pip install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-vl-4b-instruct_w4bf16_seq2048_bm1684x_1dev_20251026_141347.bmodel

# 克隆算能LLM-TPU仓库
git clone https://github.com/sophgo/LLM-TPU.git

# 复制配置文件（适配4B/8B模型）
cp -r ./LLM-TPU/models/Qwen3_VL/config/* ./models/qwen3vl_4b/

# 将下载的bmodel文件移动到对应模型目录
mv qwen3-vl-4b-instruct_w4bf16_seq2048_bm1684x_1dev_20251026_141347.bmodel ./models/qwen3vl_4b/
```

#### 1.2 （可选）手动编译模型
若需自定义模型参数，可按以下步骤编译bmodel（目前只支持在x86主机进行模型编译）：
```bash
# 1. 下载原始模型（ModelScope）
# 下载4B模型
pip install modelscope
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir Qwen3-VL-4B-Instruct

# 2. 启动算能编译容器
docker pull sophgo/tpuc_dev:latest # 若下载失败可以选下面方式下载镜像
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/24/06/14/12/sophgo-tpuc_dev-v3.2_191a433358ad.tar.gz
docker load -i sophgo-tpuc_dev-v3.2_191a433358ad.tar.gz

docker run --privileged --name qwen3vl_compile -v $PWD:/workspace -it sophgo/tpuc_dev:latest

# 3. 安装TPU-MLIR
pip install tpu_mlir # 若安装失败可以按照如下编译
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # 激活环境变量
./build.sh  # 编译mlir

# 4. 编译生成bmodel（容器内执行）
# 编译4B模型（max_input_length=1024, 768x768像素, w4bf16量化）
# 如果有提示transformers/torch版本问题，pip3 install transformers torchvision -U
llm_convert.py -m /workspace/Qwen3-VL-4B-Instruct  -s 2048 \
  --max_input_length 1024  --quantize w4bf16  -c bm1684x \
  --out_dir /workspace/qwen3vl_service/models/qwen3vl_4b  --max_pixels 768,768
# 编译完成后，在指定目录qwen3vl_4b生成qwen3-vl-xxx.bmodel和config，拷贝到算力盒子对应目录
```

#### 1.4 编译Python扩展库（必要步骤）
```bash
# 编译库文件生成chat.cpython*.so
mkdir build && cd build
cmake .. && make
cp *cpython* ../ && cd ..
```

### 2. 启动服务
#### 2.1 基础启动（使用默认参数，默认API Key: abc@123）
```bash
# 启动4B模型（默认配置，启用API Key认证）
python main_serving.py -m ./models/qwen3vl_4b
```

#### 2.2 自定义参数启动（含API Key配置）
```bash
# 示例：2B模型 + 端口9000 + 最大并发15 + 自定义API Key + 视频采样0.3
python main_serving.py \
  -m ./models/qwen3vl_2b \
  -p 9000 \
  -c 15 \
  -l DEBUG \
  -d 0 \
  -v 0.3 \
  --api-key "sk_your_custom_key_123" \
  --api-header "X-API-Key"
```

#### 2.3 后台运行
```bash
# 后台启动4B模型并输出日志（使用默认API Key）
nohup python main_serving.py -m ./models/qwen3vl_4b > service.log 2>&1 &

# 后台启动并自定义API Key
nohup python main_serving.py -m ./models/qwen3vl_4b --api-key "your_secure_key" > service.log 2>&1 &

# 查看实时日志
tail -f service.log
```

### 3. 服务参数说明
| 参数 | 简写 | 默认值 | 说明                                 |
|------|------|--------|------------------------------------|
| `--model_dir` | `-m` | `./models/qwen3vl_2b` | 模型目录路径（核心参数，2B/4B对应`./models/qwen3vl_2b`/`4b`） |
| `--max_concurrent` | `-c` | `10` | 最大并发请求数（2B建议10-15，4B建议5-10）        |
| `--log_level` | `-l` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL） |
| `--devid` | `-d` | `0` | TPU设备ID（BM1684X/BM1688设备编号）        |
| `--video_ratio` | `-v` | `0.5` | 视频采样比例（0-1，适配12秒视频/1帧/秒限制）         |
| `--port` | `-p` | `8899` | 服务端口号                              |
| `--api-key` | - | `abc@123` | API访问密钥，用于接口认证，默认值`abc@123` |
| `--api-header` | - | `Authorization` | 传递API Key的HTTP请求头名称，默认`Authorization` |
| `--api-prefix` | - | `Bearer` | API Key前缀，格式为「前缀 + 空格 + 密钥」，默认`Bearer` |

### 4. 测试服务
#### 4.1 基础健康检查（携带API Key）
```bash
# 方式1：使用默认请求头（Authorization: Bearer abc@123）
curl http://localhost:8899/health \
  -H "Authorization: Bearer abc@123"

# 方式2：若自定义了API请求头（如X-API-Key）
curl http://localhost:8899/health \
  -H "X-API-Key: abc@123"
```

#### 4.2 并发性能测试（含API Key参数）
```bash
# 测试：5并发 × 5个测试用例，使用默认API Key（abc@123）
python test_api.py -u http://0.0.0.0:8899 -c 5 -n 5

# 自定义测试参数：8并发 × 15用例 + 自定义API Key + 静默模式
python test_api.py \
  -u http://localhost:9000 \
  -c 8 \          # 8个并发请求（适配2B模型）
  -n 15 \         # 总共15个测试用例
  -t 60 \         # 超时时间60秒
  -s \            # 静默模式，仅输出到日志
  --api-key "sk_your_custom_key_123" \  # （新增）自定义API Key
  --api-header "X-API-Key" \            # （新增）自定义API请求头
  --api-prefix "ApiKey"                 # （新增）自定义API Key前缀
```

#### 4.3 验证模型运行
```bash
# 运行基础demo验证模型（可选）
python3 pipeline.py -m ./models/qwen3vl_4b/qwen3-vl-4b-instruct_w4bf16_seq2048_bm1684x_1dev_20251026_141347.bmodel -c ./models/qwen3vl_4b/config
```

#### 4.4 测试参数说明
| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--concurrent` | `-c` | `10` | 并发请求数（4B建议≤20，8B建议≤10） |
| `--timeout` | `-t` | `30` | 单个请求超时时间（秒，8B模型建议延长至60） |
| `--cases` | `-n` | `10` | 测试用例总数 |
| `--url` | `-u` | `http://0.0.0.0:8899` | API地址 |
| `--silent` | `-s` | - | 静默模式（仅输出到日志） |
| `--api-key` | - | `abc@123` | API访问密钥，与服务端配置保持一致 |
| `--api-header` | - | `Authorization` | 传递API Key的HTTP请求头名称 |
| `--api-prefix` | - | `Bearer` | API Key前缀，格式为「前缀 + 空格 + 密钥」 |

### 5. 环境适配说明
- **硬件兼容**：支持BM1684X TPU设备（SE7盒子）
- **Python版本**：推荐Python3.10，其他版本需手动适配依赖
- **依赖要求**：必须安装`torchvision`/`transformers`/`qwen_vl_utils`以保证多模态处理正常
- **API认证**：默认启用API Key认证，所有请求需携带合法密钥，测试环境可通过`--disable-api-auth`禁用

## 📁 项目结构

```
qwen3vl_service/
├── main_serving.py         # FastAPI服务主文件（核心，含API Key认证逻辑）
├── test_api.py             # 并发测试脚本（支持API Key参数）
├── pipeline.py             # 模型推理管道（编译后生成扩展）
├── build/                  # 编译目录（手动创建）
├── models/                 # 模型目录
│   ├── qwen3vl_2b/         # 2B模型文件（手动创建）
│   └── qwen3vl_4b/         # 4B模型文件（手动创建）
│   └── ...
├── chat.cpython*.so        # .so文件（编译后生成）
├── service.log             # 服务日志（运行后生成）
├── concurrent_test.log     # 测试日志（运行后生成，含API认证信息）
├── test.jpg                # 测试图片
├── test.mp4                # 测试视频
└── README.md               # 本文档
```

## 🔌 API接口

### 1. 健康检查（需携带API Key）
```bash
# 默认请求头格式
curl http://localhost:8899/health \
  -H "Authorization: Bearer abc@123"

# 自定义请求头格式（如X-API-Key）
curl http://localhost:8899/health \
  -H "X-API-Key: abc@123"
```
**响应示例**:
```json
{
  "status": "healthy",
  "details": "模型已加载且运行正常",
  "model": "qwen3-vl-instruct",
  "model_dir": "./models/qwen3vl_2b",
  "max_concurrent": 10,
  "api_auth_enabled": true,
  "timestamp": 1763000000,
  "version": "2.2.0"
}
```

### 2. 聊天对话 (OpenAI兼容，需携带API Key)
#### 纯文本对话
```bash
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer abc@123" \  # 携带API Key认证
  -d '{
    "model": "qwen3-vl-instruct",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下自己"}
    ]
  }'
```

#### 本地图片理解
```bash
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer abc@123" \  # 携带API Key认证
  -d '{
    "model": "qwen3-vl-instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这张图片"},
        {
          "type": "image_url",
          "image_url": {
            "url": "file:///path/to/your/image.jpg"
          }
        }
      ]
    }]
  }'
```

#### URL图片/视频理解
```bash
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer abc@123" \  # 携带API Key认证
  -d '{
    "model": "qwen3-vl-instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这个视频的内容"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/your-video.mp4"
          }
        }
      ]
    }]
  }'
```

#### 流式响应
```bash
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer abc@123" \  # 携带API Key认证
  -d '{
    "model": "qwen3-vl-instruct",
    "messages": [
      {"role": "user", "content": "写一段关于人工智能的短文"}
    ],
    "stream": true
  }'
```

### 3. 媒体文件上传接口（需携带API Key）
```bash
# 上传图片并描述（携带API Key）
curl -X POST http://localhost:8899/v1/media/describe \
  -H "Authorization: Bearer abc@123" \  # 携带API Key认证
  -F "file=@/path/to/your/image.jpg" \
  -F "prompt=详细描述这张图片的内容"
```

### 4. 模型信息查询（需携带API Key）
```bash
# 列出所有可用模型
curl http://localhost:8899/v1/models \
  -H "Authorization: Bearer abc@123"

# 获取指定模型详情
curl http://localhost:8899/v1/models/qwen3-vl-instruct \
  -H "Authorization: Bearer abc@123"
```

### 5. 交互式API文档
访问以下地址查看完整API文档并测试（测试接口时需手动输入API Key）：
```
http://localhost:8899/docs
```

## 🎯 核心特性

- ✅ **多并发支持** - 基于线程池的多并发处理（可配置最大并发数）
- ✅ **多模型支持** - 兼容2B/4B模型，通过`-m`参数切换
- ✅ **多模态处理** - 支持图片、视频、纯文本输入
- ✅ **媒体来源多样化**
  - 本地文件（绝对路径/相对路径/file://协议）
  - Base64编码图片
  - 远程URL（图片/视频自动下载）
- ✅ **OpenAI兼容** - 完全兼容OpenAI ChatCompletion API格式
- ✅ **流式响应** - 支持SSE流式输出
- ✅ **视频采样** - 可配置视频采样比例优化推理速度
- ✅ **线程隔离** - 每个线程独立的模型实例，避免请求干扰
- ✅ **资源自动清理** - 临时文件自动删除，避免磁盘占用
- ✅ **完善的错误处理** - 详细的日志和错误提示
- ✅ **API Key认证** - （新增）默认启用API Key保护，支持自定义请求头和前缀，提升服务安全性
- ✅ **脱敏日志** - （新增）API Key在日志中脱敏展示，避免密钥泄露

## 📊 性能参数

| 模型 | 首次加载时间 | 首Token延迟 | Token生成速度    | 最大并发  | 上下文长度 | API认证开销 |
|------|--------------|-------------|--------------|-------|------------|------------|
| 2B   | ~40秒        | ~1.2秒      | ~18 tokens/秒 | 10-15 | 2048       | 可忽略（<1ms） |
| 4B   | ~60秒        | ~1.8秒      | ~12 tokens/秒 | 5-10  | 2048       | 可忽略（<1ms） |

## 🛠️ 故障排查

### 1. 服务启动失败
```bash
# 检查模型文件是否存在
ls -lh models/qwen3vl_2b/*.bmodel

# 检查编译是否成功
ls -l *cpython*.so

# 查看详细错误日志（含API认证相关错误）
grep -i "api\|auth" service.log
grep -i error service.log
```

### 2. 模型加载失败
```bash
# 检查TPU设备状态
bm-smi

# 验证模型文件完整性
md5sum models/qwen3vl_4b/*.bmodel
```

### 3. 并发请求异常（认证相关）
```bash
# 1. 检查API Key是否一致
# 服务端配置的API Key与测试脚本是否匹配

# 2. 检查请求头格式是否正确
# 默认格式：Authorization: Bearer abc@123

# 3. 临时禁用API认证排查问题
python main_serving.py -m ./models/qwen3vl_4b -c 10 --disable-api-auth

# 4. 查看认证相关错误日志
grep -i "401\|unauthorized" service.log
```

### 4. 媒体文件处理失败
```bash
# 检查文件权限
ls -l /path/to/your/media/file.mp4

# 检查网络连接（URL媒体）
curl -I https://example.com/your-image.jpg
```

## 📋 支持的媒体格式

### 图片格式
- JPG/JPEG, PNG, BMP, GIF, WEBP

### 视频格式
- MP4, AVI, MOV, MKV, FLV, WMV

### 媒体来源
- 本地文件路径：`/absolute/path.jpg`, `./relative/path.mp4`, `../parent/path.png`
- File协议：`file:///absolute/path.jpg`
- Base64编码：`data:image/jpeg;base64,/9j/4AAQSkZJRgABA...`
- 远程URL：`http://example.com/image.jpg`, `https://example.com/video.mp4`

## 📝 注意事项

1. **模型加载**：服务启动时会预加载模型，首次请求无需等待
2. **并发设置**：4B模型建议降低并发数（5-10），避免TPU资源不足
3. **视频处理**：高分辨率视频建议降低采样比例（0.3-0.5）
4. **临时文件**：所有临时文件存储在系统临时目录，会自动清理
5. **端口占用**：确保指定端口未被占用，可通过`-p`参数修改
6. **日志级别**：调试时使用`-l DEBUG`查看详细执行过程（含API认证细节）
7. **TPU设备**：确保devid正确（默认0），可通过`sophon-smi`查看设备状态
8. **API认证**：
   - 默认API Key为`abc@123`，生产环境请修改为强密钥（包含大小写、数字、特殊字符）
   - 所有接口请求必须携带合法API Key，否则返回401未授权错误
   - 避免在命令行直接暴露API Key，生产环境建议通过环境变量传递
   - 测试环境可通过`--disable-api-auth`参数禁用认证，方便调试

## 📄 许可证

本项目基于算能官方LLM-TPU示例代码以及Qwen3-VL官方仓库开发，遵循原项目许可协议。

## 📞 技术支持

- 算能开发者社区：https://www.sophgo.com/curriculum/index.html
- SOPHON SDK文档：https://developer.sophgo.com/site/index/material/all/all.html
- 算能LLM-TPU仓库：https://github.com/sophgo/LLM-TPU
- Qwen3-VL官方仓库：https://github.com/QwenLM/Qwen3-VL

---

**更新时间**: 2025-12-16
**版本**: 0.1.0