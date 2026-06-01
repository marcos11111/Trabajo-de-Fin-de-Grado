import os
import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import paramiko
from scp import SCPClient
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

@dataclass
class RemoteConfig:
    host: str
    user: str
    password: str
    mumax_path: str
    remote_base_dir: str

class RemoteSimulator:
    def __init__(self, config: RemoteConfig, base_mx3_template: Path, local_base_results: Path, max_concurrent: int = 4):
        self.config = config
        self.base_mx3_template = Path(base_mx3_template)
        self.local_base_results = Path(local_base_results)
        self.semaphore = threading.Semaphore(max_concurrent)
        
        if not self.base_mx3_template.exists():
            raise FileNotFoundError(f"mx3 template file missing at: {self.base_mx3_template}")

    @classmethod
    def from_env(cls, base_dir: Path, local_base_results: Path, max_concurrent: int = 4):
        """Alternative constructor to bootstrap credentials and clean up system boilerplate."""
        env_path = base_dir / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            logger.info("System credentials matched and loaded out of secure environment layer.")
        else:
            logger.warning(".env configuration array missing. Relying on fallback machine variables.")
            
        try:
            config = RemoteConfig(
                host=os.environ["REMOTE_HOST"],
                user=os.environ["REMOTE_USER"],
                password=os.environ["REMOTE_PASSWORD"],
                mumax_path=os.environ["MUMAX_PATH"],
                remote_base_dir=os.environ["REMOTE_BASE_DIR"]
            )
        except KeyError as e:
            logger.critical(f"Infrastructure Deficit: Missing environment variable validation token {e}.")
            raise SystemExit(f"Please define {e} within your system environment registry or local .env file.")
            
        return cls(
            config=config,
            base_mx3_template=base_dir / "modules" / "base.mx3",
            local_base_results=local_base_results,
            max_concurrent=max_concurrent
        )

    def _run_single_sim(self, variables_dict: dict, sub_folder: str, parent_path: str, v_name: str):
        """Executes simulation remotely, transfers auxiliary files, and cleans up the remote server."""
        
        logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Esperando semáforo de concurrencia...")
        with self.semaphore:
            logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Semáforo adquirido. Inicializando cliente Paramiko...")
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            try:
                logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Conectando a {self.config.host}...")
                ssh.connect(self.config.host, username=self.config.user, password=self.config.password, timeout=30)
                logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Conexión SSH establecida con éxito.")
                
                variant_filename = f"sim_{sub_folder}.mx3"
                remote_work_dir = f"{self.config.remote_base_dir}{parent_path}/{v_name}/{sub_folder}/"
                local_dest_dir = self.local_base_results / parent_path / v_name / sub_folder
                local_dest_dir.mkdir(parents=True, exist_ok=True)
                local_mx3_path = local_dest_dir / variant_filename

                remote_work_dir_win = remote_work_dir.replace("/", "\\")
                
                logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Ejecutando 'mkdir' remoto: {remote_work_dir_win}")
                mkdir_cmd = f'if not exist "{remote_work_dir_win}" mkdir "{remote_work_dir_win}"'
                stdin, stdout, stderr = ssh.exec_command(mkdir_cmd)
                stdout.read() # Drenaje rápido
                stderr.read()
                stdout.channel.recv_exit_status()
                logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Directorio remoto confirmado.")

                local_vars = variables_dict.copy()
                files_to_upload = []

                if local_vars.get("AnisMapFile"):
                    local_vars["use_ovf_map"] = "true"
                else:
                    local_vars["use_ovf_map"] = "false"

                for var_name, value in list(local_vars.items()):
                    if isinstance(value, str) and value.lower().endswith('.ovf'):
                        ovf_path = Path(value)
                        if ovf_path.exists():
                            logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Archivo OVF local detectado para subida: {ovf_path.name} ({ovf_path.stat().st_size / 1e6:.2f} MB)")
                            files_to_upload.append((str(ovf_path), ovf_path.name))
                            local_vars[var_name] = f'"{ovf_path.name}"'
                        else:
                            raise FileNotFoundError(f"Required OVF file not found: {ovf_path}")

                with open(self.base_mx3_template, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for var_name, value in local_vars.items():
                    pattern = rf'(\b{var_name}\b\s*(?::=|=)\s*)[^/\n;]+'
                    content = re.sub(pattern, lambda m, v=value: f"{m.group(1)}{v}", content)
                
                with open(local_mx3_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                remote_file_path = remote_work_dir + variant_filename

                logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Iniciando túnel SCP...")
                with SCPClient(ssh.get_transport()) as scp:
                    scp.put(str(local_mx3_path), remote_file_path)
                    logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Script .mx3 subido.")
                    for local_ovf, ovf_name in files_to_upload:
                        logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Subiendo mapa binario {ovf_name} (Esto puede tardar un poco)...")
                        scp.put(local_ovf, remote_work_dir + ovf_name)
                        logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Mapa {ovf_name} subido con éxito.")
                
                # ⚡ FIX: Añadimos "> remote_log.txt 2>&1" para evitar el Pipe Buffer Deadlock
                run_cmd = f'cd /d "{remote_work_dir_win}" && "{self.config.mumax_path}" -gpu 2 "{variant_filename}" > remote_log.txt 2>&1'
                logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Lanzando GPU Solver con silenciador (evitando bloqueos TCP): {run_cmd}")
                
                stdin, stdout, stderr = ssh.exec_command(run_cmd)
                
                logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Script corriendo en GPU. Esperando señal de finalización remota...")
                # Como hemos redirigido la consola a un archivo, recv_exit_status ya no se congelará.
                exit_status = stdout.channel.recv_exit_status() 
                logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> GPU Solver ha finalizado. Exit Status: {exit_status}")

                if exit_status == 0:
                    remote_out_dir = remote_file_path.replace(".mx3", ".out")
                    logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Iniciando descarga de resultados (.out) por SCP...")
                    with SCPClient(ssh.get_transport()) as scp:
                        scp.get(remote_out_dir, str(local_dest_dir), recursive=True)
                    logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Descarga SCP completada.")
                    
                    local_out_temp = local_dest_dir / f"sim_{sub_folder}.out"
                    if local_out_temp.exists():
                        for file in local_out_temp.iterdir():
                            shutil.move(str(file), str(local_dest_dir / file.name))
                        local_out_temp.rmdir()
                    
                    local_j_file = local_dest_dir / "j000000.ovf"
                    if local_j_file.exists():
                        local_j_file.rename(local_dest_dir / "anisU000000.ovf")

                    remote_file_win = remote_file_path.replace("/", "\\")
                    remote_out_win = remote_out_dir.replace("/", "\\")
                    
                    # ⚡ FIX: Agregamos el archivo de log (remote_log.txt) a la cadena de limpieza
                    cleanup_cmd = f'del /q "{remote_file_win}" "{remote_work_dir_win}remote_log.txt" && rmdir /s /q "{remote_out_win}"'
                    for _, ovf_name in files_to_upload:
                        cleanup_cmd += f' && del /q "{remote_work_dir_win}{ovf_name}"'
                    
                    logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Limpiando archivos temporales en servidor...")
                    stdin, stdout, stderr = ssh.exec_command(cleanup_cmd)
                    stdout.read()
                    stderr.read()
                    stdout.channel.recv_exit_status()
                    logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Limpieza remota completada.")
                else:
                    # En caso de error, el output ahora está en el archivo log remoto en lugar de en la variable stdout
                    logger.error(f"[SSH-TRACE ERROR] {v_name}/{sub_folder} falló. El servidor arrojó el status {exit_status}.")

            except Exception as e:
                logger.error(f"[SSH-TRACE FATAL] Network/SSH Error for {v_name}/{sub_folder}: {e}")
            finally:
                logger.info(f"[SSH-TRACE] {v_name}/{sub_folder} -> Desconectando SSH y liberando semáforo.")
                ssh.close()

    def run_batch(self, parent_folder_name: str, variants: list[tuple[str, dict]], just_yes: bool = False):
        threads = []
        for v_name, var_set in variants:
            v_name_safe = v_name.strip().replace(" ", "_").replace(",", "")
            
            t_yes = threading.Thread(target=self._run_single_sim, args=(var_set, "yes", parent_folder_name, v_name_safe))
            threads.append(t_yes)
            
            if not just_yes:
                var_set_no = var_set.copy()
                var_set_no["myKu1"] = 0
                t_no = threading.Thread(target=self._run_single_sim, args=(var_set_no, "no", parent_folder_name, v_name_safe))
                threads.append(t_no)

        for t in threads:
            t.start()
            time.sleep(0.1)

        for t in threads:
            t.join()