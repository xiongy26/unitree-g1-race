// 比赛状态机: 就绪 -> 倒计时 -> 比赛 -> 结算, 含摔倒判定/扶起/罚时与排名。

import { FINISH_X } from './scene.js';

const RACE_TIMEOUT = 90;      // 秒, 超时按 DNF
const FALL_PENALTY = 2.0;     // 秒, 每次摔倒罚时
const FALL_REST = 1.2;        // 秒, 摔倒后原地扶起延时
const POST_FINISH_CMD = 1.5;  // 完赛后保持前进的秒数

export class Race {
  constructor(robots, sim, onState) {
    this.robots = robots; // { sim, runner, cmd, ...元数据 }
    this.sim = sim;
    this.onState = onState; // (state)=>void
    this.rng = null;
    this.state = 'ready';
    this.raceClock = 0;
    this.countdown = 0;
    this.placeCounter = 1;
  }

  resetAll(seed) {
    this.rng = mulberry32(seed);
    for (const r of this.robots) {
      this.sim.resetRobot(r.sim, r.laneY, this.rng, r.noise ?? 0.025, 0);
      r.runner?.reset();
      r.onRestand && r.onRestand(0);
      r.curVx = 0;
      r.finished = false;
      r.finishTime = 0;
      r.penalty = 0;
      r.falls = 0;
      r.fallen = false;
      r.fallenAt = 0;
      r.x = 0;
      r.speed = 0;
      r.place = 0;
    }
    this.state = 'ready';
    this.raceClock = 0;
    this.countdown = 0;
    this.placeCounter = 1;
    this.onState && this.onState(this.state);
  }

  start() {
    if (this.robots.length === 0) return;
    this.resetAll(this.seed ?? 1);
    this.state = 'countdown';
    this.countdown = 3.999;
    this.onState && this.onState(this.state);
  }

  newSeed(seed) { this.seed = seed >>> 0; }

  // 每个 POLICY_DT 调用一次
  tick(dt) {
    if (this.state === 'countdown') {
      this.countdown -= dt;
      if (this.countdown <= 1.0) {
        this.state = 'racing';
        this.raceClock = 0;
        // 发枪后由 stepOnce 的斜率限幅把 curVx 从 0 平滑加速到 targetSpeed,
        // 不做指令阶跃(阶跃易造成起步踉跄 -> 起跑斜向冲出车道)
        this.onState && this.onState(this.state);
      }
    }
    if (this.state !== 'racing') return;

    this.raceClock += dt;
    for (const r of this.robots) {
      const q = r.sim.data.qpos;
      r.x = q[0];
      r.speed = r.sim.data.qvel[0];

      // 完赛判定
      if (!r.finished && r.x >= FINISH_X) {
        r.finished = true;
        r.finishTime = this.raceClock + r.penalty;
        r.place = this.placeCounter++;
        r.finishedAt = this.raceClock;
      }
      if (r.finished && this.raceClock - r.finishedAt > POST_FINISH_CMD) {
        r.targetSpeed = 0; // 冲线后缓缓停下(stepOnce 的斜率限幅负责减速)
      }

      // 摔倒判定与扶起(阈值随物种; 阈值<=0 表示该物种不判摔)
      if (!r.fallen && r.fallZ > 0 && q[2] < r.fallZ && !r.finished) {
        r.fallen = true;
        r.fallenAt = this.raceClock;
        r.falls += 1;
        r.penalty += FALL_PENALTY;
      }
      if (r.fallen && this.raceClock - r.fallenAt >= FALL_REST) {
        this.sim.resetRobot(r.sim, r.laneY, this.rng, r.noise ?? 0.02, Math.max(r.x, 0));
        r.runner?.reset();
        r.onRestand && r.onRestand(Math.max(r.x, 0));
        r.curVx = r.targetSpeed;
        r.fallen = false;
      }
    }

    if (this.raceClock >= RACE_TIMEOUT || this.robots.every((r) => r.finished)) {
      this.state = 'finished';
      for (const r of this.robots) r.targetSpeed = 0;
      this.onState && this.onState(this.state);
    }
  }

  // 展示排序: 完赛者按时间在前, 未完赛按里程
  standings() {
    return [...this.robots].sort((a, b) => {
      if (a.finished && b.finished) return a.finishTime - b.finishTime;
      if (a.finished) return -1;
      if (b.finished) return 1;
      return b.x - a.x;
    });
  }
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export { RACE_TIMEOUT, FALL_PENALTY };
