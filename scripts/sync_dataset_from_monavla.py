"""
MoNaVLA → MoNa-pi 데이터셋 동기화

MoNaVLA(ROS_action/mobile_vla_dataset_v5/)가 실제 로봇 데이터 수집의 source of
truth. MoNa-pi(mobile_vla_dataset_merged/)는 그걸 학습에 쓰는 소비자 쪽이라
두 디렉토리가 시간이 지나며 어긋난다(예: 6/7 신규 에피소드 1건을 MoNa-pi가
못 받음, MoNa-pi 쪽에만 있는 구버전 4건 등 — docs/MONAVLA_CROSSCHECK_20260623.md
이후 발견).

이 스크립트는 단방향(MoNaVLA → MoNa-pi)만 동기화한다:
    - MoNaVLA에만 있는 .h5 → MoNa-pi에 symlink 생성 (복사 대신 symlink로
      디스크 중복 방지, 기존 mobile_vla_dataset_merged/의 관례와 동일)
    - MoNa-pi에만 있는 .h5는 절대 자동 처리하지 않음 — 단순 리포트만.
      (구버전/로컬 전용 데이터일 수 있어 자동 삭제·역동기화는 의도적으로 안 함)
    - 새로 symlink한 파일은 h5py로 한번 열어서 즉시 깨진 파일이면 경고

실행:
    python scripts/sync_dataset_from_monavla.py            # dry-run (기본)
    python scripts/sync_dataset_from_monavla.py --apply     # 실제 symlink 생성
"""

import argparse
from pathlib import Path

import h5py

SOURCE_DIR = Path("/home/minum/26CS/MoNaVLA/ROS_action/mobile_vla_dataset_v5")
TARGET_DIR = Path("/home/minum/26CS/MoNa-pi/mobile_vla_dataset_merged")


def list_h5(directory: Path) -> dict[str, Path]:
    return {p.name: p for p in directory.glob("*.h5")}


def check_readable(path: Path) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except Exception as e:
        print(f"  [경고] {path.name} 손상됨: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="실제로 symlink 생성 (기본은 dry-run)")
    args = parser.parse_args()

    source = list_h5(SOURCE_DIR)
    target = list_h5(TARGET_DIR)

    missing_in_target = sorted(set(source) - set(target))
    only_in_target = sorted(set(target) - set(source))

    print(f"[Sync] MoNaVLA: {len(source)}개 / MoNa-pi: {len(target)}개")
    print(f"[Sync] MoNaVLA→MoNa-pi 동기화 필요: {len(missing_in_target)}개")
    for name in missing_in_target:
        src_path = source[name].resolve()
        dst_path = TARGET_DIR / name
        if args.apply:
            dst_path.symlink_to(src_path)
            ok = check_readable(dst_path)
            print(f"  {'✅' if ok else '❌'} symlink 생성: {name}")
        else:
            print(f"  (dry-run) {name} -> {src_path}")

    if only_in_target:
        print(f"\n[Sync] MoNa-pi에만 있는 파일 (자동 처리 안 함, 수동 확인 필요): {len(only_in_target)}개")
        for name in only_in_target:
            print(f"  {name}")

    if not args.apply and missing_in_target:
        print("\n[Sync] dry-run 완료. 실제 적용하려면 --apply 추가")


if __name__ == "__main__":
    main()
