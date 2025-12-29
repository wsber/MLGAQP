import os
import subprocess


class FastestRunner:
    """
    封装对 C++ 程序 Fastest 的调用。
    支持实时输出、超时控制、参数传递等。
    """

    def __init__(self,
                 build_dir="/home/wangshuo/projects/FaSTest-main/build",
                 exe_name="Fastest"):
        """
        :param build_dir: Fastest 可执行文件所在目录
        :param exe_name:  可执行文件名称（默认 Fastest）
        """
        self.build_dir = build_dir
        self.exe_path = os.path.join(build_dir, exe_name)

        if not os.path.exists(self.exe_path):
            raise FileNotFoundError(f"❌ 找不到可执行文件: {self.exe_path}")
        if not os.access(self.exe_path, os.X_OK):
            raise PermissionError(f"⚠️ 没有执行权限，请运行: chmod +x {self.exe_path}")

    def run(self,
            dataset="parler",
            root_label=-1,
            sample_budget=None,
            estimate_with_predicate=False,  # +++ 新增参数：是否启用谓词检查 +++
            extra_args=None,
            timeout=None):
        """
        运行 Fastest 程序，并实时输出执行日志。

        :param dataset: 数据集名称 (对应参数 -d)
        :param root_label: ROOT_LABEL 值 (对应推理节点的标签)
        :param sample_budget: 采样预算
        :param estimate_with_predicate: 是否开启谓词检查模式 (True/False)
        :param extra_args: 额外的命令行参数 (list)
        :param timeout: 最大执行时间（秒）
        :return: (returncode, stdout_str)
        """
        # 基础参数
        args = [self.exe_path, "-d", dataset, "--ROOT_LABEL", str(root_label)]

        # 参数：采样预算
        if sample_budget is not None:
            args.extend(["--SAMPLE_BUDGET", str(sample_budget)])

        # +++ 参数：启用谓词检查 +++
        if estimate_with_predicate:
            args.append("--ESTIMATE_WITH_PREDICATE")

        # 其他额外参数
        if extra_args:
            args.extend(extra_args)

        print(f"🚀 正在运行: {' '.join(args)}\n")

        # 启动进程
        proc = subprocess.Popen(
            args,
            cwd=self.build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        output_lines = []
        try:
            for line in proc.stdout:
                print(line, end="")  # 实时打印
                output_lines.append(line)

            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"\n⏰ 超时（超过 {timeout} 秒），已终止进程。")

        print(f"\n✅ Fastest 运行结束，退出码: {proc.returncode}")
        return proc.returncode, "".join(output_lines)