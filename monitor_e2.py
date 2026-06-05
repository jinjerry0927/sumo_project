"""E2 검지기 실시간 모니터 — 관측함수(e2_observation.py)와 동일한 getter로
차로별 [대기열(jam veh), 차량수, 정지수, occupancy%]를 주기마다 콘솔 테이블 + CSV로 출력.
E2 기능/성능 검증 + 발표 자료용. 직접 실행.
  예) python monitor_e2.py --scenario saturated --gui --seconds 600 --interval 10"""
import os, sys, csv, argparse
os.environ.setdefault("SUMO_HOME", r"C:/Program Files (x86)/Eclipse/Sumo")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci, sumolib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="medium", help="scenarios/eval/<name>.rou.xml")
    ap.add_argument("--gui", action="store_true", help="SUMO-GUI 동시 표시")
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--interval", type=int, default=10, help="집계/출력 주기(초)")
    ap.add_argument("--delay", type=int, default=200, help="GUI step 지연(ms), --gui 일 때만")
    ap.add_argument("--csv", default="results/e2_monitor.csv")
    a = ap.parse_args()

    route = f"scenarios/eval/{a.scenario}.rou.xml"
    binary = sumolib.checkBinary("sumo-gui" if a.gui else "sumo")
    traci.start([binary, "-n", "network/intersection.net.xml", "-r", route,
                 "-a", "network/e2.add.xml", "--no-step-log", "--no-warnings"]
                + (["--start", "--quit-on-end", "--delay", str(a.delay)] if a.gui else []))

    dets = sorted(traci.lanearea.getIDList())  # e2_<laneID>
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    f = open(a.csv, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["time", "detector", "lane", "jam_veh", "veh_num", "halting", "occupancy"])

    step = 0
    while step < a.seconds:
        try:
            traci.simulationStep()
        except (traci.exceptions.FatalTraCIError, KeyboardInterrupt):
            print("\n[중단] GUI 종료/인터럽트 — 지금까지 수집분 저장")
            break
        step += 1
        if step % a.interval == 0:
            print(f"\n=== t={step:>4}s ===  {'detector':<12}{'queue':>6}{'veh':>5}{'halt':>5}{'occ%':>7}", flush=True)
            for d in dets:
                lane = traci.lanearea.getLaneID(d)
                q = traci.lanearea.getJamLengthVehicle(d)
                n = traci.lanearea.getLastStepVehicleNumber(d)
                h = traci.lanearea.getLastStepHaltingNumber(d)
                o = traci.lanearea.getLastStepOccupancy(d)
                print(f"{'':<19}{d:<12}{q:>6}{n:>5}{h:>5}{o:>7.1f}", flush=True)
                w.writerow([step, d, lane, q, n, h, f"{o:.2f}"])
            f.flush()
    try:
        traci.close()
    except Exception:
        pass
    f.close()
    print(f"\n[OK] CSV 저장: {a.csv}  ({step // a.interval} samples x {len(dets)} detectors)")

if __name__ == "__main__":
    main()
