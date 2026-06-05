"""network/e2.add.xml 생성 — TLS 진입 차로마다 E2(laneAreaDetector) 1개.
정지선(차로 끝) 앞 최대 150m 커버. ID 규약 'e2_'+차로ID (e2_observation 과 동일).
차로가 150m 미만이면 전체 커버. 새 네트워크에도 그대로 동작(차로 자동 탐색)."""
import os, sys
os.environ.setdefault("SUMO_HOME", r"C:/Program Files (x86)/Eclipse/Sumo")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib

NET, OUT, COVER = "network/intersection.net.xml", "network/e2.add.xml", 150.0
net = sumolib.net.readNet(NET)
lanes = set()
for tls in net.getTrafficLights():
    for conn in tls.getConnections():
        lanes.add(conn[0].getID())

rows = ['<additional>']
for lane in sorted(lanes):
    L = net.getLane(lane).getLength()
    length = min(COVER, L - 0.1)
    pos = max(0.0, L - length)
    rows.append(f'    <laneAreaDetector id="e2_{lane}" lane="{lane}" '
                f'pos="{pos:.2f}" length="{length:.2f}" freq="100000" '
                f'file="e2_detectors_output.xml"/>')
rows.append('</additional>')
with open(OUT, "w") as f:
    f.write("\n".join(rows) + "\n")
print(f"[OK] {OUT} - {len(lanes)} detectors")
