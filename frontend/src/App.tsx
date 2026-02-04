import { useState } from 'react';

// ===== 类型定义 =====
// 预测接口返回的数据结构（与后端响应保持一致）
interface PredictResult {
  label: number;
  probability: number;
  model_details: Record<string, number>;
  used_models: Record<string, number>;
  figures?: string[];
  failed_models?: Record<string, string>;
}

// ===== 主组件 =====
function App() {
  // ===== 基础配置 =====
  // API基础地址（可手动修改，便于切换环境）
  const [apiBase, setApiBase] = useState('http://127.0.0.1:8000');

  // ===== 必填字段 =====
  // 这里用字符串存储输入值，提交时再转换为数字
  const [fi, setFi] = useState('0.25');
  const [age, setAge] = useState('65');
  const [gender, setGender] = useState('1');

  // ===== 可选字段（JSON） =====
  // 用户可以在这里补充其他字段（如Marital、EducationLevel等）
  const [extraJson, setExtraJson] = useState('');

  // ===== 控制项 =====
  const [returnViz, setReturnViz] = useState(true);

  // ===== 请求状态 =====
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);

  // ===== 提交处理 =====
  async function handleSubmit() {
    // 进入提交状态
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // 解析可选JSON字段
      let extra: Record<string, unknown> = {};
      if (extraJson.trim().length > 0) {
        extra = JSON.parse(extraJson);
      }

      // 拼接请求体（必填字段 + 可选字段 + 控制字段）
      const payload = {
        FI: Number(fi),
        Age: Number(age),
        Gender: Number(gender),
        return_viz: returnViz,
        ...extra,
      };

      // 发送请求
      const resp = await fetch(`${apiBase.replace(/\/$/, '')}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      // 处理非200响应
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`请求失败: ${resp.status} ${text}`);
      }

      // 解析结果
      const data: PredictResult = await resp.json();
      setResult(data);
    } catch (err) {
      // 捕获错误并展示
      const message = err instanceof Error ? err.message : '未知错误';
      setError(message);
    } finally {
      // 退出提交状态
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <h1>FI/CVD 预测（最小测试页）</h1>

      <section className="card">
        <h2>API 配置</h2>
        <label>
          API 地址
          <input
            type="text"
            value={apiBase}
            onChange={(e) => setApiBase(e.target.value)}
            placeholder="http://127.0.0.1:8000"
          />
        </label>
      </section>

      <section className="card">
        <h2>必填字段</h2>
        <div className="grid">
          <label>
            FI
            <input type="number" value={fi} onChange={(e) => setFi(e.target.value)} step="0.01" />
          </label>
          <label>
            Age
            <input type="number" value={age} onChange={(e) => setAge(e.target.value)} />
          </label>
          <label>
            Gender
            <input type="number" value={gender} onChange={(e) => setGender(e.target.value)} />
          </label>
        </div>
      </section>

      <section className="card">
        <h2>可选字段（JSON）</h2>
        <p className="hint">示例：{"{"Marital":1, "EducationLevel":2}"}</p>
        <textarea
          rows={6}
          value={extraJson}
          onChange={(e) => setExtraJson(e.target.value)}
          placeholder='{"Marital":1, "EducationLevel":2}'
        />
      </section>

      <section className="card">
        <h2>控制项</h2>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={returnViz}
            onChange={(e) => setReturnViz(e.target.checked)}
          />
          返回可视化资源
        </label>
      </section>

      <section className="card">
        <button disabled={loading} onClick={handleSubmit}>
          {loading ? '预测中...' : '开始预测'}
        </button>
        {error && <div className="error">{error}</div>}
      </section>

      {result && (
        <section className="card">
          <h2>预测结果</h2>
          <p>分类结果：{result.label}</p>
          <p>风险概率：{result.probability.toFixed(4)}</p>

          <h3>模型输出</h3>
          <pre>{JSON.stringify(result.model_details, null, 2)}</pre>

          <h3>模型权重</h3>
          <pre>{JSON.stringify(result.used_models, null, 2)}</pre>

          {result.failed_models && (
            <>
              <h3>失败模型</h3>
              <pre>{JSON.stringify(result.failed_models, null, 2)}</pre>
            </>
          )}

          {result.figures && result.figures.length > 0 && (
            <>
              <h3>可视化图表</h3>
              <div className="figures">
                {result.figures.slice(0, 8).map((url) => (
                  <img key={url} src={`${apiBase.replace(/\/$/, '')}${url}`} alt={url} />
                ))}
              </div>
            </>
          )}
        </section>
      )}
    </div>
  );
}

export default App;
