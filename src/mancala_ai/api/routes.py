# src/mancala_ai/api/routes.py
from flask_smorest import Blueprint
from marshmallow import Schema, fields, validate, EXCLUDE
from mancala_ai.engine.core import new_game, step, legal_actions
from mancala_ai.io.registry import current_meta, pick_action

bp = Blueprint("mancala", __name__, url_prefix="/api")

# ---------- Schemas ----------
class StateSchema(Schema):
    class Meta: unknown = EXCLUDE
    pits = fields.List(fields.List(fields.Integer()), required=True)
    stores = fields.List(fields.Integer(), required=True)
    current_player = fields.Integer(required=True, validate=validate.OneOf([0, 1]))

class MoveReqSchema(Schema):
    class Meta: unknown = EXCLUDE
    state  = fields.Nested(StateSchema, required=True)
    agent  = fields.String(load_default="dqn",
                           validate=validate.OneOf(["dqn","minimax","alpha_beta","mcts","random", "advanced"]))
    action = fields.Integer(load_default=None)  # used by /apply

class MoveRespSchema(Schema):
    action     = fields.Integer(allow_none=True)
    next_state = fields.Nested(StateSchema)
    reward     = fields.Float()
    done       = fields.Boolean()
# -----------------------------

@bp.route("/health")
@bp.response(200, Schema.from_dict({"status": fields.String(), "model": fields.Dict()})())
def health():
    return {"status": "ok", "model": current_meta()}

@bp.route("/newgame", methods=["POST"])
@bp.response(200, Schema.from_dict({"state": fields.Nested(StateSchema)})())
def newgame():
    return {"state": new_game()}

@bp.route("/apply", methods=["POST"])  # human move
@bp.arguments(MoveReqSchema)
@bp.response(200, MoveRespSchema)
def apply(req):
    a = req.get("action")
    acts = legal_actions(req["state"])
    if a is None or a not in acts:
        from flask_smorest import abort
        abort(400, message=f"Illegal or missing action. Legal: {acts}")
    ns, rew, done = step(req["state"], a)
    return {"action": a, "next_state": ns, "reward": rew, "done": done}

@bp.route("/move", methods=["POST"])   # AI move
@bp.arguments(MoveReqSchema)
@bp.response(200, MoveRespSchema)
def move(req):
    a = pick_action(req["state"], req.get("agent", "dqn"))
    ns, rew, done = step(req["state"], a)
    return {"action": a, "next_state": ns, "reward": rew, "done": done}
