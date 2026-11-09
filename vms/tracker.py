from dataclasses import dataclass, field
from collections import Counter
import math

@dataclass
class Track:
    track_id: int
    box: tuple
    center: tuple
    last_seen: float
    votes: list = field(default_factory=list)
    brand: str = "unclassified"
    brand_score: float = 0.0
    locked: bool = False
    trail: list = field(default_factory=list)

class CentroidTracker:
    def __init__(self, max_distance=90, max_lost=1.5, vote_count=3):
        self.max_distance=max_distance
        self.max_lost=max_lost
        self.vote_count=max(1,int(vote_count))
        self.tracks={}
        self.next_id=1

    @staticmethod
    def center(box):
        x1,y1,x2,y2=box
        return ((x1+x2)//2,(y1+y2)//2)

    def update(self, boxes, now):
        unmatched=set(self.tracks)
        used=set()
        pairs=[]
        for tid,tr in self.tracks.items():
            for i,box in enumerate(boxes):
                cx,cy=self.center(box)
                d=math.hypot(cx-tr.center[0],cy-tr.center[1])
                if d<=self.max_distance:
                    pairs.append((d,tid,i))
        for _,tid,i in sorted(pairs):
            if tid not in unmatched or i in used:
                continue
            unmatched.remove(tid); used.add(i)
            tr=self.tracks[tid]
            tr.box=boxes[i]; tr.center=self.center(boxes[i]); tr.last_seen=now
            tr.trail.append(tr.center); tr.trail=tr.trail[-25:]
        for i,box in enumerate(boxes):
            if i in used: continue
            tid=self.next_id; self.next_id+=1
            c=self.center(box)
            self.tracks[tid]=Track(tid,box,c,now,trail=[c])
        for tid in list(unmatched):
            if now-self.tracks[tid].last_seen>self.max_lost:
                del self.tracks[tid]
        return list(self.tracks.values())

    def add_brand_vote(self, track, brand, score):
        if track.locked or not brand:
            return
        track.votes.append((brand,float(score)))
        track.votes=track.votes[-max(self.vote_count*3,9):]
        counts=Counter(name for name,_ in track.votes)
        winner,count=counts.most_common(1)[0]
        if count>=self.vote_count:
            matching=[s for n,s in track.votes if n==winner]
            track.brand=winner
            track.brand_score=sum(matching)/len(matching)
            track.locked=True
