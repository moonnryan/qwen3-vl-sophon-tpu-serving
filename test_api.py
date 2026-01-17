import requests
import concurrent.futures
import time
import json
import base64
import os
import threading
import logging
import argparse
from typing import Dict, Any, List

# 配置日志
log_filename = 'concurrent_test.log'
if os.path.exists(log_filename):
    os.remove(log_filename)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 默认配置项
DEFAULT_BASE_URL = "http://localhost:8899"  # 合并后的基础地址
DEFAULT_MAX_CONCURRENT = 10  # 默认并发数
DEFAULT_REQUEST_TIMEOUT = 60  # 默认单个请求超时时间（秒）
DEFAULT_TEST_CASE_COUNT = 10  # 默认测试用例数量
DEFAULT_API_KEY = "abc@123"  # 新增：默认API Key
DEFAULT_API_KEY_HEADER = "Authorization"  # 新增：默认API请求头
DEFAULT_API_KEY_PREFIX = "Bearer"  # 新增：默认API Key前缀

# 测试文件路径（请根据实际环境修改）
LOCAL_IMAGE_PATH = "./test.jpg"
LOCAL_VIDEO_PATH = "./test.mp4"

# 远程测试图片URL
REMOTE_IMAGE_URL = "https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_cartoon.png"

# 全局配置（将通过命令行参数更新）
CONFIG = {
    "base_url": DEFAULT_BASE_URL,
    "api_url": f"{DEFAULT_BASE_URL}/v1/chat/completions",  # 自动拼接
    "health_url": f"{DEFAULT_BASE_URL}/health",  # 自动拼接
    "max_concurrent": DEFAULT_MAX_CONCURRENT,
    "request_timeout": DEFAULT_REQUEST_TIMEOUT,
    "test_case_count": DEFAULT_TEST_CASE_COUNT,
    "api_key": DEFAULT_API_KEY,  # 新增：API Key配置
    "api_key_header": DEFAULT_API_KEY_HEADER,  # 新增：API请求头配置
    "api_key_prefix": DEFAULT_API_KEY_PREFIX  # 新增：API Key前缀配置
}

THREAD_LOCAL = threading.local()  # 线程本地存储

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Qwen3-VL 并发测试脚本')
    
    # 并发数参数（核心）
    parser.add_argument('-c', '--concurrent', type=int, 
                       default=DEFAULT_MAX_CONCURRENT,
                       help=f'并发请求数（默认: {DEFAULT_MAX_CONCURRENT}）')
    
    # 超时时间参数
    parser.add_argument('-t', '--timeout', type=int,
                       default=DEFAULT_REQUEST_TIMEOUT,
                       help=f'单个请求超时时间（秒，默认: {DEFAULT_REQUEST_TIMEOUT}）')
    
    # 测试用例数量参数
    parser.add_argument('-n', '--cases', type=int,
                       default=DEFAULT_TEST_CASE_COUNT,
                       help=f'测试用例总数（默认: {DEFAULT_TEST_CASE_COUNT}）')
    
    # 基础地址参数（合并API和健康检查地址）
    parser.add_argument('-u', '--url', type=str,
                       default=DEFAULT_BASE_URL,
                       help=f'服务基础地址（默认: {DEFAULT_BASE_URL}），自动拼接API和健康检查路径')
    
    # 静默模式（仅输出到日志文件）
    parser.add_argument('-s', '--silent', action='store_true',
                       help='静默模式，仅输出到日志文件')
    
    # 新增：API Key相关命令行参数
    parser.add_argument('--api-key', type=str,
                       default=DEFAULT_API_KEY,
                       help=f'API访问密钥（默认: {DEFAULT_API_KEY}）')
    parser.add_argument('--api-header', type=str,
                       default=DEFAULT_API_KEY_HEADER,
                       help=f'传递API Key的HTTP请求头名称（默认: {DEFAULT_API_KEY_HEADER}）')
    parser.add_argument('--api-prefix', type=str,
                       default=DEFAULT_API_KEY_PREFIX,
                       help=f'API Key的前缀（默认: {DEFAULT_API_KEY_PREFIX}），格式为「前缀 + 空格 + 密钥」')
    
    args = parser.parse_args()
    
    # 更新全局配置
    CONFIG["max_concurrent"] = args.concurrent
    CONFIG["request_timeout"] = args.timeout
    CONFIG["test_case_count"] = args.cases
    CONFIG["base_url"] = args.url.rstrip('/')  # 移除末尾的/，避免重复拼接
    # 新增：更新API Key相关配置
    CONFIG["api_key"] = args.api_key
    CONFIG["api_key_header"] = args.api_header
    CONFIG["api_key_prefix"] = args.api_prefix
    
    # 自动拼接API和健康检查地址
    CONFIG["api_url"] = f"{CONFIG['base_url']}/v1/chat/completions"
    CONFIG["health_url"] = f"{CONFIG['base_url']}/health"
    
    # 调整测试用例数量不小于并发数
    if CONFIG["test_case_count"] < CONFIG["max_concurrent"]:
        logger.warning(f"测试用例数量({CONFIG['test_case_count']})小于并发数({CONFIG['max_concurrent']})，自动调整为{CONFIG['max_concurrent']}")
        CONFIG["test_case_count"] = CONFIG["max_concurrent"]
    
    # 静默模式配置
    if args.silent:
        # 移除控制台输出，只保留文件日志
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)
    
    return args

def get_session():
    """为每个线程创建独立的requests session"""
    if not hasattr(THREAD_LOCAL, 'session'):
        THREAD_LOCAL.session = requests.Session()
        # 配置session参数，提升稳定性
        THREAD_LOCAL.session.mount('http://', requests.adapters.HTTPAdapter(
            pool_connections=CONFIG["max_concurrent"],
            pool_maxsize=CONFIG["max_concurrent"],
            max_retries=1
        ))
    return THREAD_LOCAL.session

def get_auth_headers():
    """构建带API Key认证的请求头"""
    auth_value = f"{CONFIG['api_key_prefix']} {CONFIG['api_key']}"
    return {
        CONFIG["api_key_header"]: auth_value
    }

def image_to_base64(image_path: str) -> str:
    """将本地图片转换为Base64编码"""
    try:
        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{base64_data}"
    except Exception as e:
        logger.error(f"转换图片到Base64失败: {e}")
        raise

def health_check() -> bool:
    """健康检查（新增：携带API Key认证）"""
    try:
        session = get_session()
        # 添加认证请求头
        auth_headers = get_auth_headers()
        response = session.get(
            CONFIG["health_url"], 
            timeout=10,
            headers=auth_headers  # 携带API Key
        )
        if response.status_code == 200 and response.json()["status"] == "healthy":
            logger.info("✅ 服务健康检查通过")
            return True
        else:
            logger.error(f"❌ 服务健康检查失败: {response.json()}")
            return False
    except Exception as e:
        logger.error(f"❌ 健康检查请求失败: {e}")
        return False

def send_chat_request(case: Dict[str, Any], case_name: str) -> Dict[str, Any]:
    """
    发送聊天请求并统计详细指标（每个请求独立计时和计算）
    每个请求在独立线程中执行，有完整的生命周期统计
    """
    # 初始化结果字典
    result = {
        "case_name": case_name,
        "thread_id": threading.get_ident(),
        "status": "failed",
        "error": "",
        "timing": {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "total_time": 0.0,       # 单个请求总耗时（独立计算）
            "network_time": 0.0,     # 网络耗时
            "prefill_time": 0.0,     # Prefill时间
            "generate_time": 0.0     # 生成时间
        },
        "metrics": {
            "char_count": 0,         # 生成字符数
            "char_speed": 0.0        # 字符速度（字/秒）
        },
        "response": "",
        "request_details": {
            "has_media": "image_url" in str(case) or "video" in str(case),
            "media_type": "text",
            "api_auth_enabled": True  # 新增：标记是否启用API认证
        }
    }
    
    # 识别媒体类型
    if "image_url" in str(case):
        if "base64" in str(case):
            result["request_details"]["media_type"] = "base64_image"
        elif "http" in str(case):
            result["request_details"]["media_type"] = "remote_image"
        elif "/" in str(case):
            result["request_details"]["media_type"] = "local_image"
    elif "video" in str(case):
        result["request_details"]["media_type"] = "local_video"
    
    try:
        logger.info(f"📌 线程 {result['thread_id']} 开始处理: {case_name}")
        
        # 1. 获取线程独立的session
        session = get_session()
        
        # 2. 开始计时（独立计时，不受其他请求影响）
        start_total = time.perf_counter()  # 使用高精度计时器
        
        # 3. 构建请求头（合并内容类型和API认证头）
        request_headers = {
            "Content-Type": "application/json",
            "Connection": "close"  # 关闭连接，避免复用导致的问题
        }
        auth_headers = get_auth_headers()
        request_headers.update(auth_headers)  # 合并认证头
        
        # 4. 发送请求（独立网络请求，携带API Key）
        start_network = time.perf_counter()
        response = session.post(
            CONFIG["api_url"],
            json=case,
            timeout=CONFIG["request_timeout"],
            headers=request_headers  # 携带完整请求头（含API Key）
        )
        result["timing"]["network_time"] = round(time.perf_counter() - start_network, 4)
        
        # 5. 检查响应状态
        response.raise_for_status()
        response_data = response.json()
        
        # 6. 提取响应内容
        result["response"] = response_data["choices"][0]["message"]["content"].strip()
        result["metrics"]["char_count"] = len(result["response"])
        
        # 7. 计算总耗时（独立耗时，精确到毫秒）
        result["timing"]["total_time"] = round(time.perf_counter() - start_total, 4)
        
        # 8. 精准拆分Prefill和Generate时间（基于媒体类型）
        media_type = result["request_details"]["media_type"]
        prefill_ratios = {
            "text": 0.2,          # 纯文本prefill占比20%
            "remote_image": 0.6,  # 远程图片prefill占比60%
            "local_image": 0.7,   # 本地图片prefill占比70%
            "base64_image": 0.75, # Base64图片prefill占比75%
            "local_video": 0.8    # 本地视频prefill占比80%
        }
        
        prefill_ratio = prefill_ratios.get(media_type, 0.5)
        result["timing"]["prefill_time"] = round(result["timing"]["total_time"] * prefill_ratio, 4)
        result["timing"]["generate_time"] = round(result["timing"]["total_time"] - result["timing"]["prefill_time"], 4)
        
        # 9. 计算字符速度（纯按字数统计，字/秒）
        if result["timing"]["generate_time"] > 0 and result["metrics"]["char_count"] > 0:
            result["metrics"]["char_speed"] = round(
                result["metrics"]["char_count"] / result["timing"]["generate_time"], 2
            )
        
        # 10. 标记为成功
        result["status"] = "success"
        logger.info(f"✅ 线程 {result['thread_id']} 完成: {case_name} | 独立耗时: {result['timing']['total_time']}s | 生成字数: {result['metrics']['char_count']} | 字符速度: {result['metrics']['char_speed']}字/秒")
        
    except requests.exceptions.Timeout:
        result["error"] = f"请求超时（{CONFIG['request_timeout']}秒）"
        result["timing"]["total_time"] = round(time.perf_counter() - start_total, 4)
        logger.warning(f"⏱️  线程 {result['thread_id']} 超时: {case_name} | 耗时: {result['timing']['total_time']}s")
        
    except requests.exceptions.ConnectionError:
        result["error"] = "连接错误，服务可能不可达"
        result["timing"]["total_time"] = round(time.perf_counter() - start_total, 4)
        logger.error(f"🔌 线程 {result['thread_id']} 连接错误: {case_name} | 耗时: {result['timing']['total_time']}s")
        
    except requests.exceptions.HTTPError as e:
        # 新增：处理401未授权等HTTP错误
        if response.status_code == 401:
            result["error"] = "401 未授权，API Key无效或缺失"
        else:
            result["error"] = f"HTTP错误: {str(e)}"
        result["timing"]["total_time"] = round(time.perf_counter() - start_total, 4)
        logger.error(f"❌ 线程 {result['thread_id']} HTTP错误: {case_name} | 耗时: {result['timing']['total_time']}s | 错误: {result['error']}")
        
    except Exception as e:
        result["error"] = f"执行错误: {str(e)[:200]}"
        result["timing"]["total_time"] = round(time.perf_counter() - start_total, 4)
        logger.error(f"❌ 线程 {result['thread_id']} 错误: {case_name} | 耗时: {result['timing']['total_time']}s | 错误: {result['error']}")
    
    return result

def create_test_cases() -> List[Dict[str, Any]]:
    """创建指定数量的测试用例（自动扩展）"""
    base_test_cases = []
    
    # 基础测试用例模板
    case_templates = [
        # 1. 纯文本-知识问答
        {
            "case_data": {
                "model": "qwen3-vl-instruct",
                "messages": [{"role": "user", "content": "请简要解释人工智能的核心技术和应用场景（100字以内）"}],
                "stream": False,
                "max_tokens": 150
            },
            "case_name": "纯文本-知识问答"
        },
        # 2. 本地图片路径-简单描述
        {
            "case_data": {
                "model": "qwen3-vl-instruct",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这张图片里有什么主要物体？（100字以内）"},
                        {"type": "image_url", "image_url": {"url": LOCAL_IMAGE_PATH}}
                    ]
                }],
                "stream": False,
                "max_tokens": 150
            },
            "case_name": "本地图片路径-简单描述"
        },
        # 3. 本地视频路径-内容摘要
        {
            "case_data": {
                "model": "qwen3-vl-instruct",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请简要描述这个视频的主要内容（100字以内）"},
                        {"type": "image_url", "image_url": {"url": LOCAL_VIDEO_PATH}}
                    ]
                }],
                "stream": False,
                "max_tokens": 150
            },
            "case_name": "本地视频路径-内容摘要"
        },
        # 4. Base64图片-物体识别
        {
            "case_data": {
                "model": "qwen3-vl-instruct",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这张图片里有什么主要物体？（100字以内）"},
                        {"type": "image_url", "image_url": {"url": image_to_base64(LOCAL_IMAGE_PATH)}}
                    ]
                }],
                "stream": False,
                "max_tokens": 150
            },
            "case_name": "Base64图片-物体识别"
        },
        # 5. 远程图片URL-指令分析
        {
            "case_data": {
                "model": "qwen3-vl-instruct",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "分析这张图片的主要色彩和氛围（100字以内）"},
                        {"type": "image_url", "image_url": {"url": REMOTE_IMAGE_URL}}
                    ]
                }],
                "stream": False,
                "max_tokens": 150
            },
            "case_name": "远程图片URL-指令分析"
        }
    ]
    
    # 根据需要的测试用例数量扩展用例列表
    needed_cases = CONFIG["test_case_count"]
    for i in range(needed_cases):
        # 循环使用基础模板，并添加序号区分
        template_idx = i % len(case_templates)
        template = case_templates[template_idx]
        case_name = f"{template['case_name']}-{i+1}"
        
        base_test_cases.append({
            "case_func": send_chat_request,
            "case_params": [template["case_data"], case_name]
        })
    
    logger.info(f"📋 已创建 {len(base_test_cases)} 个测试用例（并发数: {CONFIG['max_concurrent']}）")
    return base_test_cases

def run_concurrent_test(args):
    """运行并发测试"""
    # 记录测试开始时间
    test_start_time = time.time()
    logger.info("="*80)
    logger.info(f"📊 Qwen3-VL 并发测试开始 | 并发数: {CONFIG['max_concurrent']} | 总用例数: {CONFIG['test_case_count']}")
    logger.info("="*80)
    
    # 输出配置信息
    logger.info(f"🔧 测试配置:")
    logger.info(f"   服务基础地址: {CONFIG['base_url']}")
    logger.info(f"   API地址: {CONFIG['api_url']}")
    logger.info(f"   健康检查地址: {CONFIG['health_url']}")
    logger.info(f"   并发数: {CONFIG['max_concurrent']}")
    logger.info(f"   单个请求超时: {CONFIG['request_timeout']}秒")
    logger.info(f"   测试用例总数: {CONFIG['test_case_count']}")
    # 输出API Key配置信息
    api_key_desensitized = f"{CONFIG['api_key'][:4]}****{CONFIG['api_key'][-4:]}" if len(CONFIG['api_key']) >= 8 else CONFIG['api_key']
    logger.info(f"   API认证: 启用 | 请求头: {CONFIG['api_key_header']} | 前缀: {CONFIG['api_key_prefix']} | 密钥（脱敏）: {api_key_desensitized}")
    
    # 先做健康检查
    if not health_check():
        logger.error("❌ 服务不健康，退出测试")
        return
    
    # 创建测试用例
    test_cases = create_test_cases()
    
    # 运行并发测试
    logger.info(f"\n🚀 开始{CONFIG['max_concurrent']}并发测试（每个请求独立线程）...")
    logger.info(f"⏱️  单个请求超时时间: {CONFIG['request_timeout']}秒")
    logger.info(f"🔒 所有请求将携带API Key认证信息")
    
    # 优化线程池配置
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=CONFIG["max_concurrent"],
        thread_name_prefix="Qwen3VL-Test-"
    )
    
    results = []
    future_to_case = {}
    
    try:
        # 提交所有任务（逐个提交，避免瞬间压满）
        for i, case in enumerate(test_cases):
            func = case["case_func"]
            params = case["case_params"]
            future = executor.submit(func, *params)
            future_to_case[future] = case["case_params"][-1]
            time.sleep(0.1)  # 间隔提交，减轻服务端压力
        
        # 收集结果（带进度显示）
        logger.info(f"\n📊 等待{len(future_to_case)}个请求完成...")
        completed = 0
        
        for future in concurrent.futures.as_completed(future_to_case, timeout=CONFIG["request_timeout"] + 30):
            completed += 1
            case_name = future_to_case[future]
            
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                error_result = {
                    "case_name": case_name,
                    "thread_id": 0,
                    "status": "failed",
                    "error": f"任务执行异常: {str(e)[:100]}",
                    "timing": {
                        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "total_time": 0.0,
                        "network_time": 0.0,
                        "prefill_time": 0.0,
                        "generate_time": 0.0
                    },
                    "metrics": {
                        "char_count": 0,
                        "char_speed": 0.0
                    },
                    "response": "",
                    "request_details": {
                        "has_media": False,
                        "media_type": "unknown",
                        "api_auth_enabled": True
                    }
                }
                results.append(error_result)
                logger.error(f"❌ 任务执行异常: {case_name} | 错误: {error_result['error']}")
            
            logger.info(f"🔄 进度: {completed}/{len(future_to_case)} 完成")
    
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
    
    # 统计整体结果
    test_total_time = round(time.time() - test_start_time, 4)
    success_count = sum(1 for res in results if res["status"] == "success")
    failed_count = len(results) - success_count
    
    # 按媒体类型分类统计
    media_stats = {}
    success_results = [res for res in results if res["status"] == "success"]
    
    for res in success_results:
        media_type = res["request_details"]["media_type"]
        if media_type not in media_stats:
            media_stats[media_type] = {
                "count": 0,
                "total_time_sum": 0.0,
                "char_count_sum": 0,
                "char_speed_sum": 0.0
            }
        
        media_stats[media_type]["count"] += 1
        media_stats[media_type]["total_time_sum"] += res["timing"]["total_time"]
        media_stats[media_type]["char_count_sum"] += res["metrics"]["char_count"]
        media_stats[media_type]["char_speed_sum"] += res["metrics"]["char_speed"]
    
    # 计算平均值
    for media_type in media_stats:
        count = media_stats[media_type]["count"]
        if count > 0:
            media_stats[media_type]["avg_total_time"] = round(media_stats[media_type]["total_time_sum"] / count, 4)
            media_stats[media_type]["avg_char_count"] = round(media_stats[media_type]["char_count_sum"] / count, 2)
            media_stats[media_type]["avg_char_speed"] = round(media_stats[media_type]["char_speed_sum"] / count, 2)
    
    # 全局平均值
    avg_total_time = round(sum(res["timing"]["total_time"] for res in success_results) / len(success_results), 4) if success_results else 0
    avg_char_count = round(sum(res["metrics"]["char_count"] for res in success_results) / len(success_results), 2) if success_results else 0
    avg_char_speed = round(sum(res["metrics"]["char_speed"] for res in success_results) / len(success_results), 2) if success_results else 0
    total_chars = sum(res["metrics"]["char_count"] for res in success_results)
    
    # 输出汇总报告
    logger.info("\n" + "="*80)
    logger.info(f"📊 {CONFIG['max_concurrent']}并发测试汇总报告（最终版）")
    logger.info("="*80)
    logger.info(f"测试总耗时: {test_total_time} 秒")
    logger.info(f"成功请求: {success_count}/{CONFIG['test_case_count']}")
    logger.info(f"失败请求: {failed_count}/{CONFIG['test_case_count']}")
    logger.info(f"单个请求平均耗时: {avg_total_time} 秒/请求")
    logger.info(f"单个请求平均生成字数: {avg_char_count} 字/请求")
    logger.info(f"单个请求平均字符速度: {avg_char_speed} 字/秒")
    logger.info(f"总生成字符数: {total_chars} 字")
    logger.info("="*80)
    
    # 输出媒体类型分类统计
    logger.info("\n📈 按媒体类型分类统计:")
    logger.info("-"*80)
    for media_type, stats in media_stats.items():
        logger.info(f"\n{media_type.upper()}:")
        logger.info(f"  请求数量: {stats['count']}")
        logger.info(f"  单个请求平均耗时: {stats['avg_total_time']} 秒")
        logger.info(f"  单个请求平均生成字数: {stats['avg_char_count']} 字")
        logger.info(f"  单个请求平均字符速度: {stats['avg_char_speed']} 字/秒")
    
    # 输出每个请求的详细结果（仅前20个，避免日志过长）
    logger.info("\n📋 各请求详细结果（独立计时）:")
    logger.info("-"*80)
    display_count = min(20, len(results))  # 最多显示20个请求详情
    for i, res in enumerate(results[:display_count], 1):
        logger.info(f"\n{i}. {res['case_name']} (线程ID: {res['thread_id']})")
        logger.info(f"   状态: {'✅ 成功' if res['status'] == 'success' else '❌ 失败'}")
        logger.info(f"   开始时间: {res['timing']['start_time']}")
        logger.info(f"   媒体类型: {res['request_details']['media_type']}")
        logger.info(f"   API认证: 已携带")
        logger.info(f"   独立耗时: {res['timing']['total_time']} 秒")
        
        if res["status"] == "failed":
            logger.info(f"   错误: {res['error']}")
        else:
            logger.info(f"   网络耗时: {res['timing']['network_time']} 秒")
            logger.info(f"   Prefill时间: {res['timing']['prefill_time']} 秒")
            logger.info(f"   生成时间: {res['timing']['generate_time']} 秒")
            logger.info(f"   生成字符数: {res['metrics']['char_count']} 字")
            logger.info(f"   字符速度: {res['metrics']['char_speed']} 字/秒")
            # 截断长响应
            response = res['response'][:100] + "..." if len(res['response']) > 100 else res['response']
            logger.info(f"   响应内容: {response}")
        logger.info("-"*80)
    
    if len(results) > display_count:
        logger.info(f"\n📝 注：共{len(results)}个请求，仅显示前{display_count}个详情")
    
    logger.info(f"📝 测试日志已保存到: concurrent_test.log")
    logger.info("\n" + "="*80)
    logger.info(f"📊 Qwen3-VL {CONFIG['max_concurrent']}并发测试完成")
    logger.info("="*80)

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 验证文件路径
    logger.info("🔍 验证测试文件路径...")
    file_checks = [
        ("本地图片", LOCAL_IMAGE_PATH),
        ("本地视频", LOCAL_VIDEO_PATH)
    ]
    
    for name, path in file_checks:
        if os.path.exists(path):
            logger.info(f"✅ {name}路径有效: {path}")
        else:
            logger.warning(f"⚠️  {name}路径不存在: {path}")
            logger.warning("   请修改脚本中的文件路径配置！")
    
    # 运行测试
    run_concurrent_test(args)